#include <vector>
using namespace std;
bool afunc(const std::vector<int>& numbers, int target){
  int count = 0;
  for (int num : numbers){
    if (num == target){
      count++;
      if(count == 3){
        return true;
      }
   }
 }
  return false;
}
