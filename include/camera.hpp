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


	bool m_firstMouse = true;
	float xoffset, yoffset, lastX, lastY, m_yaw = 0.0, m_pitch = 89.0f;

	glm::mat4x4 m_projectionMatrix, m_viewMatrix;
	glm::vec3 m_posVector, m_upVector, m_frontVector, m_worldUpVector, m_rightVector;

	float4* d_projViewMatrix = { nullptr };

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

	virtual void update(MouseProps props);
	virtual void update(WindowProps props);
};
#endif
