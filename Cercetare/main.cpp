
#include "MainWindow.h"
#include "Constants.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	a.setApplicationName(QString::fromStdString(Constants::APP_NAME));
	a.setOrganizationName(QString::fromStdString(Constants::ORG_NAME));

	MainWindow w;
	w.show();
	return a.exec();
}
