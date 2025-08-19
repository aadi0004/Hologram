#version 330

uniform vec3 CamPos;
uniform float Time;
uniform float Alpha;
uniform vec3 HoloColor;
uniform float ScanFreq;
uniform float FlickerSpeed;
uniform float FresnelPower;

in vec3 v_normal_ws;
in vec3 v_pos_ws;

out vec4 fragColor;

void main() {
    // Fresnel edge glow
    vec3 N = normalize(v_normal_ws);
    vec3 V = normalize(CamPos - v_pos_ws);
    float fres = pow(1.0 - max(dot(N, V), 0.0), FresnelPower);

    // Base emission
    float base = 0.35 + 0.65 * fres;

    // Horizontal scanlines
    float scan = 0.5 + 0.5 * sin(v_pos_ws.y * ScanFreq + Time * 2.5);

    // Flicker animation
    float flicker = 0.9 + 0.1 * sin(Time * FlickerSpeed + v_pos_ws.x * 3.7);

    float emission = base * mix(0.85, 1.0, scan) * flicker;
    vec3 color = emission * HoloColor;

    fragColor = vec4(color, Alpha);
}
