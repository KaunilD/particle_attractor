#ifndef LIBS_H
#define LIBS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <exception>
#include <random>

// Qt
#include <QKeyEvent>
#include <QWidget>
#include <QTimer>
#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>
#include <QObject>
#include <QMatrix4x4>
#include <QVector3D>
#include <QWheelEvent>
#include <QVector>
#include <QFile>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QImage>
#include <QOpenGLTexture>
#include <QDialog>
#include <QMessageBox>
#include <QMainWindow>
#include <QElapsedTimer>

// OpenMP
#include <omp.h>

// GLM
#include "glm/glm.hpp"
#define LOG(x) {qDebug() << x;}

using std::shared_ptr;
using std::make_shared;

using std::unique_ptr;
using std::make_unique;

#endif