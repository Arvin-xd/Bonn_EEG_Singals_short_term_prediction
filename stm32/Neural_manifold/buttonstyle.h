#ifndef BUTTONSTYLE_H
#define BUTTONSTYLE_H

#include <QString>

class ButtonStyle
{
public:
    ButtonStyle();    

    /* 模式切换按钮样式 */
    static QString btModeInactiveStyle(void);
    static QString btModeActivetaStyle(void);

    static QString btLoadDataStyle(void);
    static QString btOpenDeviceStyle(void);

    /* 通道选择按钮样式 */
    static QString btChannelSelectInactiveStyle(void);
    static QString btChannelSelectActivetaStyle(void);
};

#endif // BUTTONSTYLE_H
