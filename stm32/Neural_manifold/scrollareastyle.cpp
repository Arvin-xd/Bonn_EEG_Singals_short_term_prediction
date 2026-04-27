

#include "scrollareastyle.h"

ScrollAreaStyle::ScrollAreaStyle()
{

}

QString ScrollAreaStyle::getChannleSelectScrollAreaStyle(){
    QString styleDescription = "scrollArea{\
                border: 2px solid #c3c3c3;\
                border-radius:5px;\
            }\
            \
            QScrollBar:vertical {\
                border: none;\
                border-radius:5px;\
                width: 10px;\
                background:rgba(198, 198, 198, 0.5);\
            }\
            QScrollBar::handle:vertical {\
                border: none;\
                border-radius:5px;\
                background-color: #044288;\
              }\
            QScrollBar::sub-line:vertical {\
                  border: none;\
                  height: 0px;\
                  subcontrol-position: top;\
                  subcontrol-origin: margin;\
              }\
            QScrollBar::add-line:vertical {\
                  border: none;\
                  height: 0px;\
                  subcontrol-position: bottom;\
                  subcontrol-origin: margin;\
              }\
              QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\
                border:none;\
                  width: 0px;\
                  height: 0px;\
              }\
              QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\
                  background: none;\
              }";
    return styleDescription;
}
