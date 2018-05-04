#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "ui_cercetare.h"
#include "GPGPUImplementation.h"

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QLabel>


class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = 0);
	~MainWindow();

	void Log(std::string str);

	virtual void wheelEvent(QWheelEvent * event);

private slots:
	void OnSwitchViewClicked();
	void OnZoomInClicked();
	void OnZoomOutClicked();
	void OnZoomResetClicked();
	void OnToggleLogClicked();

	void OnLoadImageClicked();
	void OnSaveImageClicked();
	void OnSobelClicked();
	void OnGaussianBlurClicked();
	void OnSharpeningClicked();
	void OnColorSmoothingClicked();
	void OnKMeansClicked();
	void OnGrayscaleClicked();
	void OnResizeClicked();
	void OnThresholdClicked();
	void OnSOMClicked();

	void OnSIFTClcked();

	void OnBenchmarkClicked();

private:
	Ui::CercetareClass m_ui;
	GPGPUImplementation m_cl;
	QImage m_img;
	QPixmap m_pm;
	uchar *m_img_data;

	QScrollArea *scrollAreaViewImageOriginal;
	QScrollArea *scrollAreaViewImageResult;
	QLabel *labelImageViewerOriginal;
	QLabel *labelImageViewerResult;

	double m_scale_factor;

	void LoadTexture(QString imgpath);
	void SetActions();
	void ScaleImage(double factor);
	void AdjustScrollBar(QScrollBar *scrollbar, double factor);
};

#endif // MAIN_WINDOW_H
