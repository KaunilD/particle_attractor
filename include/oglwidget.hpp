#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include "libs.hpp"
// shaderprogram
#include "shaderprogram.hpp"
// algorithms
#include "gameobject/gameobject.hpp"
#include "algorithms/algorithms.hpp"
#include "camera.hpp"
#include "renderer.hpp"

class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:

	explicit OGLWidget(QWidget *parent = 0);
	~OGLWidget();


	ShaderProgram *particleObjShader;

	Algorithms *attractor;
	std::vector<GameObject *> *gameObjects;
	QMatrix4x4 *projectionMatrix, *viewMatrix;
	Camera *camera;
	Renderer *renderer;
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int w, int h) override;
	void resize();
	bool isWall();
	void reset(unsigned int w, unsigned int h);
	void signalGameOver();
	void initializeGLfromGrid();

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