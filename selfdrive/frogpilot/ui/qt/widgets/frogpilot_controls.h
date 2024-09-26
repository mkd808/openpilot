#pragma once

#include <cmath>
#include <type_traits>

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QObject>
#include <QTimer>

#include "selfdrive/ui/qt/widgets/controls.h"

void updateFrogPilotToggles();

QColor loadThemeColors(const QString &colorKey, bool clearCache = false);

const QString buttonStyle = R"(
  QPushButton {
    padding: 0px 25px 0px 25px;
    border-radius: 50px;
    font-size: 35px;
    font-weight: 500;
    height: 100px;
    color: #E4E4E4;
    background-color: #393939;
  }
  QPushButton:pressed {
    background-color: #4a4a4a;
  }
  QPushButton:checked:enabled {
    background-color: #33Ab4C;
  }
  QPushButton:disabled {
    color: #33E4E4E4;
  }
)";

class FrogPilotConfirmationDialog : public ConfirmationDialog {
  Q_OBJECT

public:
  explicit FrogPilotConfirmationDialog(const QString &prompt_text, const QString &confirm_text,
                                       const QString &cancel_text, const bool rich, QWidget *parent);
  static bool toggle(const QString &prompt_text, const QString &confirm_text, QWidget *parent);
  static bool toggleAlert(const QString &prompt_text, const QString &button_text, QWidget *parent);
  static bool yesorno(const QString &prompt_text, QWidget *parent);
};

class FrogPilotListWidget : public QWidget {
  Q_OBJECT

public:
  explicit FrogPilotListWidget(QWidget *parent = nullptr) : QWidget(parent), outer_layout(this), inner_layout() {
    outer_layout.setMargin(0);
    outer_layout.setSpacing(0);
    outer_layout.addLayout(&inner_layout);

    inner_layout.setMargin(0);
    inner_layout.setSpacing(25);  // default spacing is 25
    outer_layout.addStretch();
  }

  inline void addItem(QWidget *w) {
    inner_layout.addWidget(w);
    adjustStretch();
  }

  inline void addItem(QLayout *layout) {
    inner_layout.addLayout(layout);
    adjustStretch();
  }

  inline void setSpacing(int spacing) {
    inner_layout.setSpacing(spacing);
  }

protected:
  void paintEvent(QPaintEvent *event) {
    QPainter p(this);
    p.setPen(Qt::gray);

    std::vector<QRect> visibleRects;
    for (int i = 0; i < inner_layout.count(); ++i) {
      QWidget *widget = inner_layout.itemAt(i)->widget();
      if (widget && widget->isVisible()) {
        visibleRects.push_back(widget->geometry());
      }
    }

    int lineOffset = inner_layout.spacing() / 2;
    for (size_t i = 0; i + 1 < visibleRects.size(); ++i) {
      int bottom = visibleRects[i].bottom() + lineOffset;
      p.drawLine(visibleRects[i].left() + 40, bottom, visibleRects[i].right() - 40, bottom);
    }
  }

private:
  void adjustStretch() {
    if (inner_layout.count() > 3) {
      outer_layout.addStretch();
    }
  }

  QVBoxLayout outer_layout;
  QVBoxLayout inner_layout;
};

class FrogPilotButtonsControl : public AbstractControl {
  Q_OBJECT

public:
  FrogPilotButtonsControl(const QString &title, const QString &desc,
                          const std::vector<QString> &buttonLabels,
                          bool checkable = false, bool exclusive = true, const QString &icon = "",
                          int minimumButtonWidth = 225, QWidget *parent = nullptr)
    : AbstractControl(title, desc, icon), buttonGroup(new QButtonGroup(this)) {

    buttonGroup->setExclusive(exclusive);

    for (int i = 0; i < buttonLabels.size(); ++i) {
      QPushButton *button = new QPushButton(buttonLabels[i], this);
      button->setCheckable(checkable);
      button->setStyleSheet(buttonStyle);
      button->setMinimumWidth(minimumButtonWidth);

      hlayout->addWidget(button);

      buttonGroup->addButton(button, i);

      connect(button, &QPushButton::clicked, this, [this, i]() { emit buttonClicked(i); });
    }
  }

  void setEnabled(bool enable) {
    for (QAbstractButton *btn : buttonGroup->buttons()) {
      btn->setEnabled(enable);
    }
  }

  void setCheckedButton(int id, bool status = true) {
    if (QAbstractButton *button = buttonGroup->button(id)) {
      button->setChecked(status);
    }
  }

  void setEnabledButtons(int id, bool enable) {
    if (QPushButton *button = qobject_cast<QPushButton *>(buttonGroup->button(id))) {
      button->setEnabled(enable);
    }
  }

  void setText(int id, const QString &text) {
    if (QPushButton *button = qobject_cast<QPushButton *>(buttonGroup->button(id))) {
      button->setText(text);
    }
  }

private:
  QButtonGroup *buttonGroup;

signals:
  void buttonClicked(int id);
};

class FrogPilotButtonToggleControl : public ParamControl {
  Q_OBJECT

public:
  FrogPilotButtonToggleControl(const QString &param, const QString &title, const QString &desc,
                               const std::vector<QString> &buttonParams, const std::vector<QString> &buttonLabels,
                               const bool exclusive = false, const int minimumButtonWidth = 225, QWidget *parent = nullptr)
    : ParamControl(param, title, desc, "", parent),
      key(param.toStdString()),
      buttonParams(buttonParams),
      buttonGroup(new QButtonGroup(this)) {

    buttonGroup->setExclusive(exclusive);

    for (int i = 0; i < buttonLabels.size(); ++i) {
      QPushButton *button = new QPushButton(buttonLabels[i], this);
      button->setCheckable(true);
      button->setStyleSheet(buttonStyle);
      button->setMinimumWidth(minimumButtonWidth);

      hlayout->addWidget(button);
      hlayout->insertWidget(hlayout->indexOf(&toggle) - 1, button);

      buttonGroup->addButton(button, i);

      connect(button, &QPushButton::clicked, this, [this, i, buttonParams](bool checked) {
        params.putBool(buttonParams[i].toStdString(), checked);
        buttonGroup->button(i)->setChecked(checked);
        emit buttonClicked(i);
      });
    }

    connect(this, &ToggleControl::toggleFlipped, this, &FrogPilotButtonToggleControl::refresh);
  }

  void showEvent(QShowEvent *event) override {
    refresh();
  }

  void refresh() {
    bool state = params.getBool(key);
    if (state != toggle.on) {
      toggle.togglePosition();
    }

    const QList<QAbstractButton *> buttons = buttonGroup->buttons();
    for (QAbstractButton *button : buttons) {
      if (button) {
        button->setVisible(state);
        int buttonIndex = buttons.indexOf(button);
        button->setChecked(params.getBool(buttonParams[buttonIndex].toStdString()));
      }
    }
  }

  void setEnabledButtons(int id, bool enable) {
    if (QPushButton *button = qobject_cast<QPushButton *>(buttonGroup->button(id))) {
      button->setEnabled(enable);
    }
  }

private:
  Params params;
  QButtonGroup *buttonGroup;
  std::string key;
  std::vector<QString> buttonParams;

signals:
  void buttonClicked(int id);
};

class FrogPilotParamManageControl : public ParamControl {
  Q_OBJECT

public:
  FrogPilotParamManageControl(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent = nullptr)
    : ParamControl(param, title, desc, icon, parent),
      key(param.toStdString()),
      manageButton(new ButtonControl("", tr("MANAGE"))) {

    hlayout->insertWidget(hlayout->indexOf(&toggle) - 1, manageButton);

    connect(this, &ToggleControl::toggleFlipped, this, &FrogPilotParamManageControl::refresh);
    connect(manageButton, &ButtonControl::clicked, this, &FrogPilotParamManageControl::manageButtonClicked);
  }

  void showEvent(QShowEvent *event) {
    ParamControl::showEvent(event);
    refresh();
  }

signals:
  void manageButtonClicked();

private:
  void refresh() {
    manageButton->setEnabled(this->isEnabled() && params.getBool(key));
  }

  ButtonControl *manageButton;
  Params params;
  std::string key;
};

class FrogPilotParamValueControl : public AbstractControl {
  Q_OBJECT

public:
  FrogPilotParamValueControl(const QString &param, const QString &title, const QString &desc, const QString &icon,
                             float minValue, float maxValue, const QString &label = "", const std::map<int, QString> &valueLabels = {},
                             float interval = 1.0f, bool compactSize = false, bool instantUpdate = false)
    : AbstractControl(title, desc, icon), key(param.toStdString()), minValue(minValue), maxValue(maxValue),
      labelText(label), interval(interval), valueLabels(valueLabels),
      decimalPlaces(std::ceil(-std::log10(interval))), factor(std::pow(10.0f, decimalPlaces)),
      instantUpdate(instantUpdate) {

    setupButton(decrementButton, "-");
    setupButton(incrementButton, "+");

    valueLabel = new QLabel(this);
    valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    valueLabel->setStyleSheet("QLabel { color: #E0E879; }");
    if (compactSize) {
      valueLabel->setFixedSize(175, 100);
    } else {
      valueLabel->setFixedSize(350, 100);
    }

    hlayout->addWidget(valueLabel);
    hlayout->addWidget(&decrementButton);
    hlayout->addWidget(&incrementButton);

    QObject::connect(&decrementButton, &QPushButton::pressed, this, &FrogPilotParamValueControl::onDecrementPressed);
    QObject::connect(&incrementButton, &QPushButton::pressed, this, &FrogPilotParamValueControl::onIncrementPressed);
    QObject::connect(&decrementButton, &QPushButton::released, this, &FrogPilotParamValueControl::onButtonReleased);
    QObject::connect(&incrementButton, &QPushButton::released, this, &FrogPilotParamValueControl::onButtonReleased);
  }

  void updateControl(float newMinValue, float newMaxValue, const QString &newLabel = "") {
    minValue = newMinValue;
    maxValue = newMaxValue;
    labelText = newLabel;
    refresh();
  }

  void refresh() {
    value = params.getFloat(key);
    updateValueDisplay();
  }

signals:
  void valueChanged(float value);

protected:
  void showEvent(QShowEvent *event) override {
    refresh();
  }

private slots:
  void onIncrementPressed() {
    adjustValue(interval);
  }

  void onDecrementPressed() {
    adjustValue(-interval);
  }

  void onButtonReleased() {
    float lastValue = value;
    QTimer::singleShot(50, this, [this, lastValue]() {
      if (lastValue != value) {
        return;
      }

      previousDelta = false;
      params.putFloat(key, value);
      if (!instantUpdate) {
        emit valueChanged(value);
      }
    });
  }

private:
  void adjustValue(float delta) {
    float modResult = fmod(value, 5.0f * interval);
    if (modResult < interval) {
      if (previousDelta) {
        delta *= 5;
      }
      previousDelta = true;
    }

    value = qBound(minValue, value + delta, maxValue);
    value = std::round(value * factor) / factor;

    updateValueDisplay();
  }

  void updateValueDisplay() {
    int intValue = static_cast<int>(value);
    if (valueLabels.count(intValue)) {
      valueLabel->setText(valueLabels.at(intValue));
    } else {
      valueLabel->setText(QString::number(value, 'f', decimalPlaces) + labelText);
    }

    if (instantUpdate) {
      emit valueChanged(value);
    }
  }

  void setupButton(QPushButton &button, const QString &text) {
    button.setFixedSize(150, 100);
    button.setText(text);
    button.setAutoRepeat(true);
    button.setAutoRepeatInterval(150);
    button.setAutoRepeatDelay(500);
    button.setStyleSheet(R"(
      QPushButton {
        border-radius: 50px;
        font-size: 50px;
        font-weight: 500;
        height: 100px;
        padding: 0 25px;
        color: #E4E4E4;
        background-color: #393939;
      }
      QPushButton:pressed {
        background-color: #4a4a4a;
      }
    )");
  }

  Params params;

  QLabel *valueLabel;

  QPushButton decrementButton;
  QPushButton incrementButton;

  QString labelText;

  bool instantUpdate;
  bool previousDelta;

  int decimalPlaces;

  float factor;
  float interval;
  float minValue;
  float maxValue;
  float value;

  std::map<int, QString> valueLabels;

  std::string key;
};

class FrogPilotParamValueButtonControl : public FrogPilotParamValueControl {
  Q_OBJECT

public:
  FrogPilotParamValueButtonControl(const QString &param, const QString &title, const QString &desc, const QString &icon,
                                   float minValue, float maxValue, const QString &label = "", const std::map<int, QString> &valueLabels = {},
                                   float interval = 1.0f,
                                   const std::vector<QString> &buttonParams = {}, const std::vector<QString> &buttonLabels = {},
                                   const bool checkable = true, const int minimumButtonWidth = 225, QWidget *parent = nullptr)
    : FrogPilotParamValueControl(param, title, desc, icon, minValue, maxValue, label, valueLabels, interval, true),
      buttonParams(buttonParams),
      buttonGroup(new QButtonGroup(this)) {

    buttonGroup->setExclusive(false);

    for (int i = 0; i < buttonLabels.size(); ++i) {
      QPushButton *button = new QPushButton(buttonLabels[i], this);
      button->setCheckable(checkable);
      button->setStyleSheet(buttonStyle);
      button->setMinimumWidth(minimumButtonWidth);

      hlayout->addWidget(button);
      buttonGroup->addButton(button, i);

      connect(button, &QPushButton::clicked, this, [this, i, buttonParams](bool checked) {
        params.putBool(buttonParams[i].toStdString(), checked);
        buttonGroup->button(i)->setChecked(checked);
        emit buttonClicked(i);
      });
    }

    connect(this, &FrogPilotParamValueControl::valueChanged, this, &FrogPilotParamValueButtonControl::refreshButtons);
  }

protected:
  void showEvent(QShowEvent *event) override {
    FrogPilotParamValueControl::showEvent(event);
    refreshButtons();
  }

private slots:
  void refreshButtons() {
    const auto buttons = buttonGroup->buttons();
    for (int i = 0; i < buttons.size(); ++i) {
      if (auto button = buttons[i]) {
        button->setChecked(params.getBool(buttonParams[i].toStdString()));
      }
    }
  }

private:
  Params params;

  QButtonGroup *buttonGroup;

  std::vector<QString> buttonParams;

signals:
  void buttonClicked(int id);
};

class FrogPilotDualParamControl : public QFrame {
  Q_OBJECT

public:
  FrogPilotDualParamControl(FrogPilotParamValueControl *control1, FrogPilotParamValueControl *control2, QWidget *parent = nullptr)
    : QFrame(parent), control1(control1), control2(control2) {
    QHBoxLayout *hlayout = new QHBoxLayout(this);
    hlayout->addWidget(control1);
    hlayout->addWidget(control2);
  }

  void updateControl(float newMinValue, float newMaxValue, const QString &newLabel = "") {
    control1->updateControl(newMinValue, newMaxValue, newLabel);
    control2->updateControl(newMinValue, newMaxValue, newLabel);
  }

  void refresh() {
    control1->refresh();
    control2->refresh();
  }

private:
  FrogPilotParamValueControl *control1;
  FrogPilotParamValueControl *control2;
};

inline void tryConnect(QObject *controlToggle, std::function<void()> slot) {
  if (!controlToggle) {
    return;
  }

  FrogPilotButtonToggleControl *buttonToggleControl = qobject_cast<FrogPilotButtonToggleControl*>(controlToggle);
  if (buttonToggleControl) {
    QObject::connect(buttonToggleControl, &FrogPilotButtonToggleControl::buttonClicked, [slot]() {
      slot();
    });
    return;
  }

  FrogPilotParamValueControl *paramValueControl = qobject_cast<FrogPilotParamValueControl*>(controlToggle);
  if (paramValueControl) {
    QObject::connect(paramValueControl, &FrogPilotParamValueControl::valueChanged, [slot]() {
      slot();
    });
    return;
  }

  ParamControl *paramControl = qobject_cast<ParamControl*>(controlToggle);
  if (paramControl) {
    QObject::connect(paramControl, &ToggleControl::toggleFlipped, [slot]() {
      slot();
    });
    return;
  }

  ToggleControl *toggleControl = qobject_cast<ToggleControl*>(controlToggle);
  if (toggleControl) {
    QObject::connect(toggleControl, &ToggleControl::toggleFlipped, [slot]() {
      slot();
    });
    return;
  }
}
