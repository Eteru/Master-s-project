#pragma once
#include <QDialog>
#include <QLabel>

class ShowImageDialog
	: public QDialog
{
	Q_OBJECT

public:
	ShowImageDialog();
	virtual ~ShowImageDialog();

	void SetImage(const QImage & img);

private:
	QLabel *m_image_label;
};

