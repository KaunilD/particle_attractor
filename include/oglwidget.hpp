#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "renderer/particlerenderer.hpp"

#include "gameobject/gameobject.hpp"
#include "gameobject/sunobject.hpp"
#include "gameobject/particleobject.hpp"

#include "algorithms/algorithms.hpp"

class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:

	explicit OGLWidget(QWidget *parent = 0);
	~OGLWidget();

	shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects;
	
	shared_ptr<ShaderProgram> particleObjShader;
	shared_ptr<Camera> camera;
	shared_ptr<ParticleRenderer> particleRenderer;

	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int w, int h) override;
	void resize();
	bool isWall();
	void reset(unsigned int w, unsigned int h);
	void signalGameOver();
	void initializeGLfromGrid();

	/* meshes and materials */
	shared_ptr<Mesh> m_sphereMesh;
	shared_ptr<Material> m_sphereMaterial;
private:
	int frame{0};
	QTimer updateGLTimer, frameTimeUpdateTimer;
	QElapsedTimer elapsedTimer;
	double elapsedTime;
	void loadShader(const char * vs, const char * fs);
signals:
	void gameOver();
	void fps(QString);
public slots:
	void frameTimeUpdateTimerTicked();
protected:
	void keyPressEvent(QKeyEvent *);
	void wheelEvent(QWheelEvent *);
};

#endif