#include <iostream>
#include <cmath>
using namespace std;
double compute(int n){
  double output = 0.0;
  for (int i = 1; i <= n; ++i ){
    if(i % 2 == 0){
    output -=1 / pow (i, 2.0);
    }else{
    output += 1 / pow (i, 2.0);
    }
  }
  return output;
}
