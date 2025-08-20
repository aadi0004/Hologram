#version 330

out vec4 fragColor;
uniform vec3 HoloColor;
uniform float Alpha;

void main() {
    // glowing effect
    float glow = 0.6 + 0.4 * sin(gl_FragCoord.y * 0.05 + gl_FragCoord.x * 0.05);
    fragColor = vec4(HoloColor * glow, Alpha);
}
