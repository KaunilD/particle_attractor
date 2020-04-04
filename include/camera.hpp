#ifndef CAMERA_H
#define CAMERA_H
#include "libs.hpp"
#include "events/updatable.hpp"

class Camera : public Updatable{
public:
	float m_fov;
	float m_near;
	float m_far;

	int m_frameBufferWidth, m_frameBufferHeight;

	double m_speed;

	glm::mat4x4 m_projectionMatrix, m_viewMatrix;
	glm::vec3 m_posVector, m_upVector, m_frontVector, m_worldUpVector, m_rightVector;

	Camera();
	Camera(
		glm::vec3 posVector, glm::vec3 frontVector, glm::vec3 upVector,
		float fov, int fbW, int fbH, float _near, float _far
	);
	~Camera();

	void setSpeed(double speed);

	void updateViewMatrix();
	void updateProjectionMatrix(int frameBufferWidth, int frameBufferHeight, float fov);
	void resetProjectionMatrix();

	glm::mat4x4 getProjectionMatrix() const;
	glm::mat4x4 getViewMatrix() const;

	void update(float, float, float);

	virtual void update(MouseProps props);
};
#endif CAMERA_H