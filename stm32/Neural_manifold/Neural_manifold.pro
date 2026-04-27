QT       += serialport
QT       += core gui widgets charts

# Qt6ºÊ»›≈‰÷√
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14
#CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    adjustwidget.cpp \
    buttonstyle.cpp \
    channelselectlayout.cpp \
    configparam.cpp \
    configparammanager.cpp \
    controldialog.cpp \
    event.cpp \
    hardwarethread.cpp \
    loaddata.cpp \
    main.cpp \
    main_widget.cpp \
    plotdialog.cpp \
    scrollareastyle.cpp \
    statuslightstyle.cpp

HEADERS += \
    adipr4/source/ADIPR4.h \
    adjustwidget.h \
    buttonstyle.h \
    channelselectlayout.h \
    configparam.h \
    configparammanager.h \
    controldialog.h \
    event.h \
    hardwarethread.h \
    loaddata.h \
    main_widget.h \
    multiplexer/source/io.h \
    multiplexer/source/usb_device.h \
    plotdialog.h \
    scrollareastyle.h \
    statuslightstyle.h

FORMS += \
    adjustwidget.ui \
    controldialog.ui \
    mainwidget.ui \
    plotdialog.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    icon.qrc

DISTFILES += \
    icon/PNG/icon_title _bar_background.png

#RC_ICONS = icon/ICO/app.ico
