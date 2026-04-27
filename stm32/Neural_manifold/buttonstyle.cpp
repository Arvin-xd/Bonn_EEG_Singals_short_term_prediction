#include "buttonstyle.h"

ButtonStyle::ButtonStyle()
{

}

/* 模式切换按钮样式 */
QString ButtonStyle::btModeInactiveStyle()
{
    QString styleDescription =
            "QPushButton {\
                font-family: HarmonyOS Sans SC;\
                border-radius: 15px;\
                font:bold 32px;\
                color: black;\
                background-color: rgb(255, 255, 255);\
                border-color: #8B7355;\
                border: 1px groove gray;\
                border-style: outset;\
            }\
            \
            QPushButton:hover {\
                font:bold 34px;\
            }\
            \
            QPushButton:pressed {\
                color: white;\
                font:bold 36px;\
                background-color: rgb(24, 82, 154);\
                border-style:inset;\
            }";
    return styleDescription;
}

QString ButtonStyle::btModeActivetaStyle()
{
    QString styleDescription =
            "QPushButton {\
                border-radius: 15px;\
                font-family: HarmonyOS Sans SC;\
                font:bold 32px;\
                color: white;\
                background-color: rgb(24, 82, 154);\
                border-color: #8B7355;\
                border: 1px groove gray;\
                border-style: outset;\
            }";
    return styleDescription;
}

QString ButtonStyle::btLoadDataStyle(void)
{
    QString styleDescription =
            "QPushButton {\
                font-family: HarmonyOS Sans SC;\
                border-top-right-radius: 15px;\
                font:bold 32px;\
                color: black;\
                background-color: rgb(255, 255, 255);\
                border-color: #8B7355;\
                border: 0px groove gray;\
                border-style: outset;\
            }\
            \
            QPushButton:hover {\
                font:bold 34px;\
            }\
            \
            QPushButton:pressed {\
                color: white;\
                font:bold 36px;\
                background-color: rgb(24, 82, 154);\
                border-style:inset;\
            }";
    return styleDescription;
}

//QString ButtonStyle::btModeManualActivetaStyle(void)
//{
//    QString styleDescription =
//            "QPushButton {\
//                border-top-right-radius: 15px;\
//                font-family: HarmonyOS Sans SC;\
//                font:bold 32px;\
//                color: white;\
//                background-color: rgb(24, 82, 154);\
//                border-color: #8B7355;\
//                border: 0px groove gray;\
//                border-style: outset;\
//            }";
//    return styleDescription;
//}

//QString ButtonStyle::btModeAutoInactiveStyle(void)
//{
//    QString styleDescription =
//            "QPushButton {\
//                font-family: HarmonyOS Sans SC;\
//                border-bottom-right-radius: 15px;\
//                font:bold 32px;\
//                color: black;\
//                background-color: rgb(255, 255, 255);\
//                border-color: #8B7355;\
//                border: 0px groove gray;\
//                border-style: outset;\
//            }\
//            \
//            QPushButton:hover {\
//                font:bold 34px;\
//            }\
//            \
//            QPushButton:pressed {\
//                color: white;\
//                font:bold 36px;\
//                background-color: rgb(24, 82, 154);\
//                border-style:inset;\
//            }";
//    return styleDescription;
//}

QString ButtonStyle::btOpenDeviceStyle(void)
{
        QString styleDescription =
                "QPushButton {\
                    font-family: HarmonyOS Sans SC;\
                    border-bottom-right-radius: 15px;\
                    font:bold 32px;\
                    color: white;\
                    background-color: rgb(24, 82, 154);\
                    border-color: #8B7355;\
                    border: 0px groove gray;\
                    border-style: outset;\
                }\
                \
                QPushButton:hover {\
                    font:bold 34px;\
                }\
                \
                QPushButton:pressed {\
                    color: black;\
                    background-color: rgb(255, 255, 255);\
                    font:bold 36px;\
                    border-style:inset;\
                }";
        return styleDescription;
}


/* 通道选择按钮样式 */
QString ButtonStyle::btChannelSelectInactiveStyle()
{
    QString styleDescription =
            "QPushButton {\
                font-family: HarmonyOS Sans SC;\
                font:14px;\
                color: black;\
                background-color: rgba(238,238,238,62);\
                border-color: #87A5C9;\
                border: 1px groove gray;\
                border-radius: 8px;\
                border-style: outset;\
            }\
            \
            QPushButton:hover {\
                font:bold 14px;\
            }\
            \
            QPushButton:pressed {\
                color: white;\
                font:bold 15px;\
                background-color: #13549B;\
                border-style:inset;\
            }";
    return styleDescription;
}

QString ButtonStyle::btChannelSelectActivetaStyle()
{
    QString styleDescription =
            "QPushButton {\
                font-family: HarmonyOS Sans SC;\
                font:bold 14px;\
                color: white;\
                background-color: #13549B;\
                border-color: #87A5C9;\
                border: 1px groove gray;\
                border-radius: 8px;\
                border-style: outset;\
            }\
            \
            QPushButton:hover {\
                font:bold 14px;\
            }\
            \
            QPushButton:pressed {\
                color: black;\
                font:bold 15px;\
                background-color: rgba(238,238,238,62);\
                border-style:inset;\
            }";
    return styleDescription;
}
