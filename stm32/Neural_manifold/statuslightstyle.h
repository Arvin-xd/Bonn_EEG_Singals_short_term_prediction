#ifndef STATUSLIGHTSTYLE_H
#define STATUSLIGHTSTYLE_H

#include <QString>

class StatusLightStyle
{
public:
    StatusLightStyle();

    static QString getChannelActivationStyle();
    static QString getChannelCloseStyle();

    static QString getAutoModeActivationStyle();
    static QString getAutoModeCloseStyle();

    static QString getPulseOutputStyle();
    static QString getPulseStopStyle();
};

#endif // STATUSLIGHTSTYLE_H
