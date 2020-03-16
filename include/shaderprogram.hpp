#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include "libs.hpp"
#include "camera.hpp"

class ShaderProgram{
	public:

		ShaderProgram();
		ShaderProgram(QObject * parent);

		QOpenGLShaderProgram * program;

		GLuint modelMatrix;
		GLuint viewMatrix;
		GLuint projectionMatrix;
		GLuint colorVec3;

		void loadShaders(const char * vs_path, const char * fs_path);
		bool activate();
		void deactivate();

		
		void sendMatricesToShader(QMatrix4x4 projectionMatrix, QMatrix4x4 viewMatrix, QMatrix4x4 modelMatrix);
		void sendColorToShader(QVector3D color);
		void sendMatricesToShader(Camera camera);
};

#endif // SHADERPROGRAM_H
