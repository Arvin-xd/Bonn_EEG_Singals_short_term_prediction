#include "hardwarethread.h"
#include "main_widget.h"

#include <QApplication>
#include <QDebug>
#include <QLibrary>
#include <QMessageBox>
#include <QThread>

#include <serial.h>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    qRegisterMetaType<QList<int>>("QList<int>");

    MainWidget w;

    w.show();

    int res = a.exec();

    return res;
}
