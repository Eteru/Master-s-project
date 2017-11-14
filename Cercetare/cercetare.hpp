#ifndef CERCETARE_H
#define CERCETARE_H

#include "ui_cercetare.h"
#include "CLWrapper.hpp"

#include <QtWidgets/QMainWindow>

class Cercetare : public QMainWindow
{
	Q_OBJECT

public:
	Cercetare(QWidget *parent = 0);
	~Cercetare();

	void Log(std::string str);

private slots:
	void OnLoadImageClicked();
	void OnSaveImageClicked();
	void OnSobelClicked();
	void OnGaussianBlurClicked();
	void OnSharpeningClicked();
	void OnColorSmoothingClicked();
	void OnKMeansClicked();
	void OnGrayscaleClicked();
	void OnThresholdClicked();
	void OnSOMClicked();

private:
	Ui::CercetareClass m_ui;
	CLWrapper m_cl;
	QImage m_img;
	QPixmap m_pm;
	uchar *m_img_data;

	void LoadTexture(QString imgpath);
};

#endif // CERCETARE_H
