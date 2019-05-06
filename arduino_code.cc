#include <Servo.h>

Servo myservo;  // create servo object to control a servo


void setup() {
  // put your setup code here, to run once:
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  while (Serial.available() > 0) {
    // look for the next valid integer in the incoming serial stream:
    int hangle = Serial.parseInt();
    int vangle = Serial.parseInt();
    if (Serial.read() == '\n') {
      myservo.write(hangle);
    }
  }
}
