#include "oglwidget.hpp"

OGLWidget::OGLWidget(QWidget *parent) : QOpenGLWidget(parent)
{
	
	QSurfaceFormat format;
	format.setSamples(16);
	format.setProfile(QSurfaceFormat::CompatibilityProfile);
	this->setFormat(format);
	this->setFocus();

	attractor = new Algorithms();
	
	updateGLTimer.setInterval(1000 / 60.f);
	connect(&updateGLTimer, SIGNAL(timeout()), this, SLOT(update()));

	frameTimeUpdateTimer.setInterval(1000);
	connect(&frameTimeUpdateTimer, SIGNAL(timeout()), this, SLOT(frameTimeUpdateTimerTicked()));
	
}


OGLWidget::~OGLWidget() {
}

void OGLWidget::initializeGL() {

	qDebug() << "initializeOGL";
	initializeOpenGLFunctions();

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	
	camera = new Camera(
		QVector3D(0.0f, 0.0f, 10.0f),
		QVector3D(0.0f, 0.0f, -1.0f),
		QVector3D(0.0f, 1.0f, 0.0f),
		90.0f, width(), height(), 0.001f, 100.f
	);
	camera->setSpeed(0.5);

	particleObjShader = make_shared<ShaderProgram>(this);
	particleObjShader->loadShaders(
		":/vertexShader",
		":/fragShader"
	);
	
	renderer = new Renderer();
	initializeGLfromGrid();
	elapsedTimer.start();
	updateGLTimer.start();
	frameTimeUpdateTimer.start();
}

void OGLWidget::frameTimeUpdateTimerTicked() {
	QString perfUpdate("");
	perfUpdate += "FPS: " + QString::number(frame) + " Frame Time: " + QString::number(elapsedTime * 0.000001);
	
	emit fps(perfUpdate);
	
	frame = 0;
	elapsedTimer.restart();
}


void OGLWidget::initializeGLfromGrid() {
	gameObjects = make_shared<vector<GameObject *>>();

	GameObject *sunObject = new GameObject(
		true, 
		100.f, 
		QVector3D(0, 0, 0),
		QVector3D(1.0f, 0.0, 1.0f)
	);

	sunObject->loadObject(
		QString(":/sphere"), 
		NULL
	);
	sunObject->setupGLBuffers();
	gameObjects->push_back(sunObject);
	
	GameObject *particleObject = new GameObject(
		false,
		attractor->randomInt(10) / 10.0f+0.01,
		QVector3D(
			attractor->randomInt(10)-5,
			attractor->randomInt(10)-5,
			attractor->randomInt(10)-5
		),
		QVector3D(0.0, 1.0, 1.0)
	);
	particleObject->loadObject(QString(":/sphere"), NULL);
	particleObject->setupGLBuffers();
	gameObjects->push_back(particleObject);

	for (int i = 0; i < 50; i++) {
		GameObject *pObj = new GameObject(
			false,
			attractor->randomInt(10) / 10.0f + 0.01,
			QVector3D(
				attractor->randomInt(10) - 5,
				attractor->randomInt(10) - 5,
				attractor->randomInt(10) - 5
			),
			QVector3D(0.0, 1.0, 1.0)
		);
		gameObjects->push_back(pObj);
	}
}


void OGLWidget::paintGL() {

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	particleObjShader->activate();
	/*
	 render sun
	*/

	particleObjShader->sendMatricesToShader(
		camera->getProjectionMatrix(),
		camera->getViewMatrix(),
		gameObjects->at(0)->getModelMatrix()
	);
	renderer->render(
		*particleObjShader,
		*gameObjects->at(0)->attributeBuffer,
		*gameObjects->at(0)->indexBuffer,
		*(gameObjects->at(0)->texture)
	);
	/*
		render stars
	*/
	for (int i = 0; i < 50; i++) {
		if (!gameObjects->at(i)->npc) {
			float force = gameObjects->at(i)->getForceVector(
				gameObjects->at(0)
			);

			gameObjects->at(i)->applyForceVector(
				force, gameObjects->at(0)->getPosition() - gameObjects->at(i)->getPosition()
			);
		}
		particleObjShader->sendMatricesToShader(
			camera->getProjectionMatrix(),
			camera->getViewMatrix(),
			gameObjects->at(i)->getModelMatrix()
		);
		renderer->render(
			*particleObjShader,
			*gameObjects->at(1)->attributeBuffer,
			*gameObjects->at(1)->indexBuffer,
			*(gameObjects->at(1)->texture)
		);
		//gameObjects->at(i)->render(particleObjShader);
		
	}
	particleObjShader->deactivate();
	
	frame += 1;
	elapsedTime = elapsedTimer.nsecsElapsed();
}

void OGLWidget::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	camera->updateProjectionMatrix(width, height);
}

void OGLWidget::reset(unsigned int w, unsigned int h) {
	qDebug() << "reset clicked" << w << " " << h;

	updateGLTimer.stop();
	
	initializeGLfromGrid();
	this->setFocus();
	updateGLTimer.start();
}

void OGLWidget::signalGameOver() {
	emit gameOver();
}

void OGLWidget::keyPressEvent(QKeyEvent * event) {
	for (int i = 0; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(frame, event, attractor);
	}
}

void OGLWidget::wheelEvent(QWheelEvent * event) {
	camera->update(event);
}