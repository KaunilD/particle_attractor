#version 440 core

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Material {
    float shininess;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
uniform Light light;

uniform sampler2D textureSampler;

uniform highp mat3 normalMat;
uniform highp vec3 cameraEye;

in vec3 vs_normal;
in vec4 vs_position;
in vec4 vs_velocity;

out vec4 gl_FragColor;

void main()
{
    gl_FragColor = vs_velocity;
}


