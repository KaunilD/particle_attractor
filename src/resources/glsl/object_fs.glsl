#version 440 core

uniform sampler2D textureSampler;

uniform highp mat3 normalMat;

uniform highp vec3 cameraEye;

uniform highp vec3 lightColor;
uniform highp vec3 lightAmbient;
uniform highp vec3 lightPosition;

uniform highp vec3 color;

in vec3 vs_normal;
in vec4 vs_position;

out vec4 gl_FragColor;

void main()
{
    /*
        ambient lighting
    */ 
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    /*
        diffuse lighting
    */
    vec3 normal = normalize(vs_normal);
    vec3 lightDirection = normalize(lightPosition - vs_position.xyz);
    float diffuseStrength = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diffuseStrength * lightColor;
    /*
        specular lighting
    */
    float specularStrength = 0.5;
    vec3 viewDirection = normalize(cameraEye - vs_position.xyz);
    vec3 reflectDirection = reflect(-lightPosition, normal);
    
    float specularValue = pow(max(dot(viewDirection, reflectDirection), 0.0), 32);
    vec3 specular = specularStrength * specularValue * lightColor;

    vec3 result = (ambient + diffuse + specular) * color;
    gl_FragColor = vec4(result, 1.0f);
}


