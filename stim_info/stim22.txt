#include <iostream>
#include <cmath>
using namespace std;
int afunc(int terms){
  if (terms == 0){
  return 1;
  } else {
  return std::pow(2, terms) +afunc(terms - 1);
  }
}
