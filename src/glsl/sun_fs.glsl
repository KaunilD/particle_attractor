#version 440 core

uniform highp vec3 color;

in vec4 vs_position;

out vec4 gl_FragColor;

void main()
{
    gl_FragColor = vec4(vec3(color), 1.0f);
}


