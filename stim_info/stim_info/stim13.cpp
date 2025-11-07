# include <iostream>
# include <vector>
using namespace std;
std::vector<int> sequence(int start, int end, int diff){
  std::vector<int> result;
  if (start > end) {
  return result;
  }else{
  while (start <= end) {
  result.push_back(end);
  end -= diff;
  }
  return result;
  }
}
  
  
