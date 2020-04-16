#include "gameobject/material.hpp"

Material::Material(const std::string& texturePath) {
    LOG("Material:: reading image");
    cv::Mat mat = cv::imread(texturePath);
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


    glTexImage2D(GL_TEXTURE_2D, // Type of texture
        0,                      // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB,                 // Internal colour format to convert to
        mat.cols,               // Image width  i.e. 640 for Kinect in standard mode
        mat.rows,               // Image height i.e. 480 for Kinect in standard mode
        0,                      // Border width in pixels (can either be 1 or 0)
        GL_BGR,                 // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,       // Image data type
        mat.ptr()               // The actual image data itself
    );

    LOG("Material:: image transfered to the GPU");

};

void Material::updateFrame(cv::Mat mat) {
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, // Type of texture
        0,                      // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB,                 // Internal colour format to convert to
        mat.cols,               // Image width  i.e. 640 for Kinect in standard mode
        mat.rows,               // Image height i.e. 480 for Kinect in standard mode
        0,                      // Border width in pixels (can either be 1 or 0)
        GL_BGR,                 // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,       // Image data type
        mat.ptr()               // The actual image data itself
    );
    glBindTexture(GL_TEXTURE_2D, 0);
}

