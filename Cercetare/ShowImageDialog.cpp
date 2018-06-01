
#include "ShowImageDialog.h"
#include <QBoxLayout>

ShowImageDialog::ShowImageDialog()
{
	m_image_label = new QLabel;

	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_image_label);

	setLayout(layout);

	setWindowTitle(tr("Features to find"));
	setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
}

ShowImageDialog::~ShowImageDialog()
{
}

void ShowImageDialog::SetImage(const QImage & img)
{
	m_image_label->setPixmap(QPixmap::fromImage(img));
}
