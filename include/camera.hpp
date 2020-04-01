#ifndef CAMERA_H
#define CAMERA_H

#include "libs.hpp"

enum Movement {
	W,
	A,
	S,
	D
};

class Camera {
public:
	float fov;
	float _near;
	float _far;

	int frameBufferWidth, frameBufferHeight;

	double speed;

	glm::mat4x4 projectionMatrix, viewMatrix;

	glm::vec3 posVector, upVector, frontVector;

	Camera();
	Camera(
		glm::vec3 posVector, glm::vec3 frontVector, glm::vec3 upVector,
		float fov, int fbW, int fbH, float _near, float _far
	);
	~Camera();

	void setSpeed(double speed);
	void processKB(Movement mov, float speed);

	void updateViewMatrix();
	void updateProjectionMatrix(int frameBufferWidth, int frameBufferHeight);
	void resetProjectionMatrix();

	glm::mat4x4 getProjectionMatrix() const;
	glm::mat4x4 getViewMatrix() const;

	void update();
};

#endif