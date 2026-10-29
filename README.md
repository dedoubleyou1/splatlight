# Splatlight

A browser demo of a fully relightable volumetric Gaussian-splat renderer (WebGPU): every Gaussian carries
material properties and is lit at runtime by emissive Gaussian lights, sky and sun, with bounce light, shadows
and motion blur. The initial scene is a Cornell box.

Extracted from the GS Game Engine research repository; only the real-time web renderer lives here.

## Run locally

    python3 serve.py        # http://127.0.0.1:8771/

Needs a browser with WebGPU (recent Chrome, Edge or Safari). Drag to orbit, shift- or right-drag to pan,
wheel to dolly.

## Layout

    index.html                 page
    src/                       host code (boot, renderer, lighting, UI)
    shaders/                   splat pipeline WGSL (project, sort, splat, display)
    runtime/wgsl/              lighting kernels WGSL
    runtime/assets/            the Cornell box asset (runtime/SPEC format) and environment presets
