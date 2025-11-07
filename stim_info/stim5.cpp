#include <iostream>
class Point{
  public:
  Point(){
    std:cout << "Constructor called";
  }
};

int main()
{
  Point t1, *t2;
  return 0;
}
