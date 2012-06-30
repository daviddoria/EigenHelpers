#include "EigenHelpers.h"

#include <iostream>

#include <Eigen/Dense>

void TestScaleVector();

int main( int argc, char ** argv )
{
  TestScaleVector();

  return 0;
}

void TestScaleVector()
{
  Eigen::VectorXf v(5);
  v[0] = -2;
  v[1] = -1;
  v[2] = 0;
  v[3] = 1;
  v[4] = 2;
  Eigen::VectorXf scaled = EigenHelpers::ScaleVector(v, 0.0f, 1.0f);
  for(int i = 0; i < scaled.size(); ++i)
  {
    std::cout << scaled[i] << " ";
  }
  std::cout << std::endl;
}
