// Metal Shading Language source, compiled at runtime with device.makeLibrary(source:options:)
// (no offline `metal` toolchain is needed to build the app).
//
// Struct layouts here must match the private Swift mirrors in NeuronRenderer.swift
// (NSUniforms, NSTargetParams, NSGlowParams, NSMarker, NSArrow, NSArrowParams).
//
// Slow motion: arrows are instanced screen-facing quad strips (shaft + arrowhead + a travelling dot), alpha-blended
// and crisp (no glow). Their reveal is computed on the GPU from ArrowParams.progress, so the arrow buffer is static
// per trace. While a trace is on, desks not in it dim to ~25% and trace desks take their colour from traceInfo.
//
// Look: every unit is a crisp, opaque, depth-tested round dot (4x MSAA, no bloom, no additive blending),
// so overlapping planes occlude instead of summing toward white. Activation is shown as COLOR SATURATION:
// a resting unit is a dim desaturated grey-blue; as |activation| / layerScale rises its colour moves toward a
// fully saturated hue (amber +, cyan/blue −; diff: red up, blue down) with only a modest luminance rise.
// All colours are authored in sRGB and written straight into the bgra8Unorm target.

enum Shaders {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float4x4 view;
        float4x4 proj;
        float2 viewport;      // drawable size in pixels
        float morph;          // 0 = posA, 1 = posB (already eased)
        float exposure;       // saturation gain
        float threshold;      // normalized |activation| below this stays grey
        float flare;          // 0…1 right after a live token arrives (unused for colour: no flashes)
        float pointSize;
        uint colorMode;       // 0 activation, 1 diff, 2 contrast (amber = more for A, cyan = more for B)
        uint numNeurons;
        uint numUnits;
        float minPx;          // minimum dot radius in pixels
        float fadeNear;       // view depth where the depth fade starts
        float fadeFar;        // view depth where the depth fade is full
        float traceOn;        // 1 = slow-motion trace showing: dim non-trace desks, colour trace desks from traceInfo
        float traceDim;       // brightness kept by desks not in the trace (~0.25)
        float traceMap;       // 0…1: trace-ordered column map in force (traced desks big with rings, the rest faint dust)
        float hlOn;           // 1 = lesson floor highlight in force
        uint neuronsPerLayer;
        uint headsPerLayer;
        float hlPad;
        float4 hl[16];        // per-floor strength (floor f at hl[f / 4][f % 4]): 1 full, 0 dust
    };

    struct TargetParams {
        uint numNeurons;
        uint numHeads;
        uint neuronsPerLayer;
        uint headsPerLayer;
        uint numLayers;
        uint mode;            // 0 activation, 1 diff or contrast (signed map from the diff buffer)
        uint pad0;
        uint pad1;
    };

    struct GlowParams {
        uint numUnits;
        float decay;          // 0.5^(dt/halfLife)
        uint snap;            // 1 = value = target
        uint pulse;           // 1 = rising targets jump instantly (max)
    };

    struct Marker {
        uint unit;
        uint kind;            // 0 ablated, 1 scaled, 2 pinned, 3 selection, 4 hover, 5 source outline, 6/7 trace node ring (amber/blue)
    };

    struct Arrow {
        uint from;            // source unit (flat neuron index)
        uint to;              // target unit
        float weight;         // signed, normalized to its step's strongest arrow (−1…1)
        float width;          // shaft width in points (0.5…2 → 1…4 device px on Retina)
        float start;          // progress at which it starts drawing
        float end;            // progress at which it reaches the target
        float alpha;
        uint focus;           // 1 = an arrow into the focus desk (drawn on top, never recedes)
    };

    struct ArrowParams {
        float progress;       // floors revealed so far (fractional)
        float scale;          // device pixels per point
        float pad0;
        float pad1;
    };

    // ---- Palette (sRGB) ----------------------------------------------------------------------
    constant float3 kBackground = float3(0.030, 0.036, 0.058);
    constant float3 kRest       = float3(0.125, 0.145, 0.195);   // dim desaturated grey-blue
    constant float3 kRestHead   = float3(0.170, 0.190, 0.250);
    constant float3 kPositive   = float3(1.000, 0.520, 0.040);   // saturated amber/orange
    constant float3 kNegative   = float3(0.020, 0.560, 1.000);   // saturated cyan-blue
    constant float3 kDiffUp     = float3(1.000, 0.110, 0.080);   // saturated red
    constant float3 kDiffDown   = float3(0.120, 0.300, 1.000);   // saturated blue

    // ---- Saturation curve --------------------------------------------------------------------
    // THE tuning function: normalized |activation| (1.0 ≈ the layer's 99.5th percentile) → colour
    // saturation 0…1. A smoothstep that opens at the threshold, shaped by a soft knee (power > 1) so the
    // bulk of a heavy-tailed layer stays near grey and only the top few percent reach a vivid hue.
    float saturation(float normalized, float threshold) {
        float s = smoothstep(threshold, threshold + 0.80, normalized);
        return pow(s, 1.6);
    }

    // ---- Compute: per-unit normalized signed target ------------------------------------------
    kernel void computeTargets(device const half *neurons [[buffer(0)]],
                               device const float *heads [[buffer(1)]],
                               device const float *diff [[buffer(2)]],
                               device const float *layerScales [[buffer(3)]],
                               device const float *headScales [[buffer(4)]],
                               device float *targets [[buffer(5)]],
                               constant TargetParams &p [[buffer(6)]],
                               uint i [[thread_position_in_grid]]) {
        uint total = p.numNeurons + p.numHeads;
        if (i >= total) return;
        if (i < p.numNeurons) {
            uint l = min(i / max(p.neuronsPerLayer, 1u), p.numLayers - 1);
            float s = max(layerScales[l], 1e-6);
            float v = (p.mode == 1) ? diff[i] : float(neurons[i]);
            targets[i] = v / s;
        } else {
            uint j = i - p.numNeurons;
            if (p.mode == 1) { targets[i] = 0.0; return; }
            uint l = min(j / max(p.headsPerLayer, 1u), p.numLayers - 1);
            float s = max(headScales[l], 1e-6);
            targets[i] = heads[j] / s;
        }
    }

    // ---- Compute: saturation persistence -------------------------------------------------------
    // The per-unit signed value that drives saturation. On a live pulse it jumps to a rising target and then
    // decays toward the current target with the half-life, so a unit's colour fades back to grey over time.
    kernel void updateGlow(device const float *targets [[buffer(0)]],
                           device float *glow [[buffer(1)]],
                           constant GlowParams &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= p.numUnits) return;
        float t = targets[i];
        if (p.snap != 0) { glow[i] = t; return; }
        float g = glow[i];
        float ng = mix(t, g, p.decay);              // decay toward target
        if (p.pulse != 0 && abs(t) > abs(ng)) ng = t; // pulse: max(value·decay, target)
        glow[i] = ng;
    }

    // ---- Dots --------------------------------------------------------------------------------
    struct SpriteOut {
        float4 position [[position]];
        float2 uv;
        float3 color;
    };

    constant float2 kCorners[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };

    static float4 unitPos(device const float4 *posA, device const float4 *posB, uint idx, float morph) {
        return mix(posA[idx], posB[idx], morph);   // xyz world, w = world radius
    }

    vertex SpriteOut spriteVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                  device const float4 *posA [[buffer(0)]],
                                  device const float4 *posB [[buffer(1)]],
                                  device const float *glow [[buffer(2)]],
                                  constant Uniforms &u [[buffer(3)]],
                                  device const float2 *traceInfo [[buffer(4)]]) {
        SpriteOut o;
        float4 p = unitPos(posA, posB, iid, u.morph);
        bool isHead = iid >= u.numNeurons;
        float g = glow[iid];
        float n = abs(g) * max(u.exposure, 0.0);
        // Head norms are all positive and cluster well above zero; gate them higher so only standouts colour.
        if (isHead && u.colorMode == 0) n = max(n - 0.55, 0.0) / 0.45;
        float s = saturate(saturation(n, u.threshold));

        float3 hue;
        if (u.colorMode == 1) hue = g >= 0.0 ? kDiffUp : kDiffDown;
        else hue = g >= 0.0 ? kPositive : kNegative;
        float3 col = mix(isHead ? kRestHead : kRest, hue, s);
        float traceGrow = 1.0;
        float minScale = 1.0;
        float mapMix = smoothstep(0.0, 1.0, u.traceMap);
        if (u.traceOn > 0.5) {
            // x: 0 not in the trace, 1 a target (y = signed arriving sum, −1…1), 2 a source-only desk (y = reveal 0…1).
            float2 ti = traceInfo[iid];
            if (ti.x < 0.5) {
                col = mix(kBackground, col, mix(u.traceDim, 0.18, mapMix));
                traceGrow = mix(1.0, 0.55, mapMix);          // dust: smaller and fainter, never a glow
                minScale = mix(1.0, 0.6, mapMix);
            } else if (ti.x < 1.5) {
                float fs = saturate(abs(ti.y));
                float3 th = ti.y >= 0.0 ? kPositive : kNegative;
                col = mix(kRest * 1.35, th, pow(fs, 0.8));
                traceGrow = mix(1.35, 3.0, mapMix);          // traced node: ~3x the normal dot
                minScale = mix(1.0, 3.0, mapMix);
            } else {
                col = mix(kRest * 1.35, col, saturate(ti.y));
                traceGrow = mix(1.35, 3.0, mapMix);
                minScale = mix(1.0, 3.0, mapMix);
            }
        }

        if (u.hlOn > 0.5) {
            uint fl = isHead ? (iid - u.numNeurons) / max(u.headsPerLayer, 1u) : iid / max(u.neuronsPerLayer, 1u);
            fl = min(fl, 63u);
            float hw = u.hl[fl >> 2][fl & 3u];
            col = mix(kBackground, col, mix(0.15, 1.0, hw));   // other floors: faint dust
            // Highlighted floors: resting desks lifted to a visible dim grey (busy desks keep their full saturated hue).
            // 0.5, not 1.1: at 8,192 desks a floor the resting dots cover most of the field, and a brighter grey
            // drowned the few saturated ones.
            col += kRest * 0.5 * hw * (1.0 - s);
            traceGrow *= mix(0.6, 1.0, hw);
            minScale *= mix(0.6, 1.0, hw);
        }

        float4 vp = u.view * float4(p.xyz, 1.0);
        float depth = max(-vp.z, 1e-3);
        // Depth fade: back planes sink slightly toward the background.
        float fade = saturate((depth - u.fadeNear) / max(u.fadeFar - u.fadeNear, 1e-3));
        col = mix(col, kBackground, fade * 0.6);

        // Footprint: at most a slight size increase with saturation; never a halo.
        float pxPerWorld = u.proj[1][1] / depth * u.viewport.y * 0.5;
        float natural = p.w * u.pointSize * (isHead ? 2.3 : 1.0);
        float radius = natural * (1.0 + 0.15 * s) * traceGrow;
        float px = radius * pxPerWorld;
        // The pixel floor keeps far dots visible, but for resting desks never past ~0.4 of the desk pitch on screen
        // (p.w is 0.3 pitch): at 8,192 desks a floor the pitch can be ~3 px, and a fixed floor fused neighbours into
        // solid grey sheets. Busy desks keep the full floor (a third larger at full saturation), so a saturated dot stays
        // a readable crisp disc when the field is pixel-limited; up close the floor is inactive and only the 15% applies.
        float minPxHere = mix(min(u.minPx, natural * pxPerWorld * 1.3), u.minPx * 1.35, s) * minScale;
        if (px < minPxHere) radius *= minPxHere / max(px, 1e-4);
        float2 c = kCorners[vid];
        vp.xy += c * radius;
        // Busy desks win ties with their resting neighbours: where dots overlap (dense floors, cluster islands;
        // 8,192 desks a floor) a saturated dot sits a hair nearer the eye, so its colour is never hidden behind grey.
        vp.z += s * radius * 0.6;
        o.position = u.proj * vp;
        o.uv = c;
        o.color = col;
        return o;
    }

    fragment half4 spriteFragment(SpriteOut in [[stage_in]]) {
        float d2 = dot(in.uv, in.uv);
        if (d2 > 1.0) discard_fragment();
        // Crisp disc with a faint darker rim so neighbouring dots stay separable.
        float3 c = in.color * mix(1.0, 0.78, smoothstep(0.55, 1.0, d2));
        return half4(half3(c), 1.0h);
    }

    // ---- Markers -----------------------------------------------------------------------------
    struct MarkerOut {
        float4 position [[position]];
        float2 uv;
        float4 color;
        float width;
        uint kind;
    };

    vertex MarkerOut markerVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                  device const float4 *posA [[buffer(0)]],
                                  device const float4 *posB [[buffer(1)]],
                                  device const Marker *markers [[buffer(2)]],
                                  constant Uniforms &u [[buffer(3)]]) {
        MarkerOut o;
        Marker m = markers[iid];
        uint idx = min(m.unit, u.numUnits - 1);
        float4 p = unitPos(posA, posB, idx, u.morph);
        bool isHead = idx >= u.numNeurons;
        float radius = p.w * u.pointSize * (isHead ? 2.3 : 1.0) * (m.kind >= 6 ? 4.5 : 2.6);
        float4 vp = u.view * float4(p.xyz, 1.0);
        float depth = max(-vp.z, 1e-3);
        float px = radius * u.proj[1][1] / depth * u.viewport.y * 0.5;
        float minPx = (m.kind == 3 || m.kind == 4) ? 10.0 : (m.kind == 5 ? 6.0 : 7.0);
        if (m.kind >= 6) minPx = u.minPx * 4.5;      // a ring hugging a traced node (min dot radius x3, ring at 0.82 of this)
        if (px < minPx) radius *= minPx / max(px, 1e-4);
        float2 c = kCorners[vid];
        vp.xy += c * radius;
        o.position = u.proj * vp;
        o.uv = c;
        o.kind = m.kind;
        switch (m.kind) {
            case 0: o.color = float4(0.86, 0.10, 0.12, 1.0); o.width = 0.14; break;  // ablated
            case 1: o.color = float4(0.70, 0.50, 1.00, 1.0); o.width = 0.12; break;  // scaled
            case 2: o.color = float4(1.00, 0.84, 0.30, 1.0); o.width = 0.12; break;  // pinned
            case 3: o.color = float4(0.96, 0.97, 1.00, 1.0); o.width = 0.15; break;  // selection
            case 6: o.color = float4(1.00, 0.52, 0.04, 1.0); o.width = 0.26; break;  // trace node: pushes (amber)
            case 7: o.color = float4(0.02, 0.56, 1.00, 1.0); o.width = 0.26; break;  // trace node: holds back (blue)
            case 5: o.color = float4(0.86, 0.89, 0.96, 0.85); o.width = 0.11; break; // slow-motion source outline
            default: o.color = float4(0.80, 0.85, 0.95, 0.8); o.width = 0.07; break; // hover
        }
        return o;
    }

    fragment half4 markerFragment(MarkerOut in [[stage_in]]) {
        float d = length(in.uv);
        if (d > 1.0) discard_fragment();
        float aa = 0.05;
        float ring = 1.0 - smoothstep(in.width * 0.5, in.width * 0.5 + aa, abs(d - 0.82));
        float cover = ring;
        if (in.kind == 0) {
            float l1 = abs(in.uv.x - in.uv.y) * 0.7071, l2 = abs(in.uv.x + in.uv.y) * 0.7071;
            float cross = (1.0 - smoothstep(0.06, 0.06 + aa, min(l1, l2))) * (1.0 - smoothstep(0.6, 0.68, d));
            cover = max(cover, cross);
        }
        float a = cover * in.color.a;
        if (a < 0.003) discard_fragment();
        return half4(half3(in.color.rgb), half(a));
    }

    // ---- Arrows (slow motion) ----------------------------------------------------------------
    // 15 vertices per instance, triangle list: 0–5 shaft quad, 6–8 arrowhead, 9–14 travelling dot.
    // Built in device pixels from the projected endpoints, so widths are exact and screen-facing in any layout.
    struct ArrowOut {
        float4 position [[position]];
        float2 uv;
        float4 color;
        uint kind [[flat]];   // 0 shaft/head, 1 dot
    };

    constant float2 kShaftQuad[6] = { float2(0,-1), float2(1,-1), float2(0,1), float2(0,1), float2(1,-1), float2(1,1) };
    constant float2 kDotQuad[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1) };

    vertex ArrowOut arrowVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                device const float4 *posA [[buffer(0)]],
                                device const float4 *posB [[buffer(1)]],
                                device const Arrow *arrows [[buffer(2)]],
                                constant Uniforms &u [[buffer(3)]],
                                constant ArrowParams &ap [[buffer(4)]]) {
        ArrowOut o;
        o.uv = float2(0.0);
        o.color = float4(0.0);
        o.kind = 0;
        o.position = float4(-4.0, -4.0, 0.5, 1.0);   // degenerate, off-screen: not drawn (yet)
        Arrow a = arrows[iid];
        float f = saturate((ap.progress - a.start) / max(a.end - a.start, 1e-4));
        if (f <= 0.0) return o;
        uint last = max(u.numUnits, 1u) - 1u;
        float4 pa = unitPos(posA, posB, min(a.from, last), u.morph);
        float4 pb = unitPos(posA, posB, min(a.to, last), u.morph);
        float4 ca = u.proj * (u.view * float4(pa.xyz, 1.0));
        float4 cb = u.proj * (u.view * float4(pb.xyz, 1.0));
        if (ca.w <= 0.02 || cb.w <= 0.02) return o;
        float2 sa = (ca.xy / ca.w * 0.5 + 0.5) * u.viewport;
        float2 sb = (cb.xy / cb.w * 0.5 + 0.5) * u.viewport;
        float2 d = sb - sa;
        float len = length(d);
        if (len < 1.0) return o;
        float2 dir = d / len;
        float2 nrm = float2(-dir.y, dir.x);
        float sc = max(ap.scale, 1.0);
        float tipLen = len * f;
        float2 tip = sa + dir * tipLen;
        float w = max(a.width, 1.0) * sc;
        float headL = min((3.5 + 2.0 * a.width) * sc, tipLen);
        float headW = (1.6 + 1.2 * a.width) * sc;
        float2 shaftEnd = tip - dir * headL * 0.7;

        float m = saturate(abs(a.weight));
        float3 hue = a.weight >= 0.0 ? kPositive : kNegative;
        float3 col = mix(kRest, hue, 0.45 + 0.55 * m);     // saturation rule: weaker arrows are greyer
        float alpha = a.alpha;
        // Finished steps recede so the step in flight reads: the previous one at ~60%, older ones faint.
        if (a.focus == 0) {
            float age = ap.progress - (floor(a.start) + 1.0);
            if (age >= 1.0) alpha *= 0.18; else if (age >= 0.0) alpha *= mix(0.6, 0.18, age);
        }
        float2 p;
        if (vid < 6) {
            float2 c = kShaftQuad[vid];
            p = mix(sa, shaftEnd, c.x) + nrm * (c.y * w * 0.5);
        } else if (vid < 9) {
            uint k = vid - 6;
            p = (k == 0) ? tip : (tip - dir * headL + nrm * ((k == 1) ? headW : -headW));
        } else {
            if (f >= 1.0) return o;                          // the dot travels only while the arrow grows
            float2 c = kDotQuad[vid - 9];
            float r = (0.5 * a.width + 1.5) * sc;
            p = tip + c * r;
            o.uv = c;
            o.kind = 1;
            col = mix(hue, float3(1.0), 0.4);                 // bright, opaque, no halo
            alpha = 1.0;
        }
        o.position = float4(p / u.viewport * 2.0 - 1.0, 0.5, 1.0);
        o.color = float4(col, alpha);
        return o;
    }

    fragment half4 arrowFragment(ArrowOut in [[stage_in]]) {
        if (in.kind == 1 && dot(in.uv, in.uv) > 1.0) discard_fragment();
        return half4(half3(in.color.rgb), half(in.color.a));
    }
    """
}
