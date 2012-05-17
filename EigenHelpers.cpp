#include "EigenHelpers.h"

// STL
#include <iostream>
#include <stdexcept>

namespace EigenHelpers
{

float SumOfRow(const Eigen::MatrixXf& m, const unsigned int rowId)
{
  return SumOfVector(m.row(rowId));
}

float SumOfVector(const Eigen::VectorXf& v)
{
  float sum = 0.0f;
  for(unsigned int i = 0; i < static_cast<unsigned int>(v.size()); ++i)
  {
    sum += v[i];
  }
  return sum;
}

float SumOfAbsoluteDifferences(const Eigen::VectorXf& a, const Eigen::VectorXf& b)
{
  if(a.size() != b.size())
  {
    throw std::runtime_error("SumOfAbsoluteDifferences: vectors must be the same size!");
  }

  float total = 0.0f;
  for(unsigned int i = 0; i < static_cast<unsigned int>(a.size()); ++i)
  {
    total += fabs(a[i] - b[i]);
  }

  return total;
}

std::vector<float> EigenVectorToSTDVector(const Eigen::VectorXf& vec)
{
  std::vector<float> stdvector(vec.size());
  for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); ++i)
  {
    stdvector[i] = vec[i];
  }
  return stdvector;
}

void OutputVectors(const VectorOfVectors& vectors)
{
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    std::cout << vectors[i] << std::endl;
  }
}

Eigen::MatrixXf TruncateColumns(const Eigen::MatrixXf& m, const unsigned int numberOfColumnsToKeep)
{
  Eigen::MatrixXf truncated = Eigen::MatrixXf::Zero(m.rows(), numberOfColumnsToKeep);
  for(int r = 0; r < truncated.rows(); ++r)
  {
    for(int c = 0; c < truncated.cols(); ++c)
    {
      truncated(r,c) = m(r,c);
    }
  }
  return truncated;
}

void OutputMatrixSize(const Eigen::MatrixXf& m)
{
  std::cout << m.rows() << "x" << m.cols() << std::endl;
}

}
