
#include "cercetare.hpp"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	a.setApplicationName("Placeholder");
	a.setOrganizationName("Eteru");

	Cercetare w;
	w.show();
	return a.exec();
}
