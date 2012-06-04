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


Eigen::VectorXf ComputeMeanVector(const EigenHelpers::VectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }
  // Compute mean vector
  Eigen::VectorXf meanVector = Eigen::VectorXf::Zero(vectors[0].size());
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    meanVector += vectors[i];
  }

  meanVector /= vectors.size();

  return meanVector;
}

Eigen::VectorXf ComputeMinVector(const EigenHelpers::VectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }

  Eigen::VectorXf minVector = Eigen::VectorXf::Zero(vectors[0].size());
  for(int dim = 0; dim < vectors[0].size(); ++dim) // loop through each dimension
  {
    std::vector<float> values(vectors[0].size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      values[i] = vectors[i](dim);
    }
    minVector[dim] = *(std::min_element(values.begin(), values.end()));
  }

  return minVector;
}

Eigen::VectorXf ComputeMaxVector(const EigenHelpers::VectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }

  Eigen::VectorXf maxVector = Eigen::VectorXf::Zero(vectors[0].size());
  for(int dim = 0; dim < vectors[0].size(); ++dim) // loop through each dimension
  {
    std::vector<float> values(vectors[0].size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      values[i] = vectors[i](dim);
    }
    maxVector[dim] = *(std::max_element(values.begin(), values.end()));
  }

  return maxVector;
}

Eigen::MatrixXf ConstructCovarianceMatrix(const EigenHelpers::VectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the covariance matrix of a list of vectors of length zero!");
  }

  Eigen::VectorXf meanVector = ComputeMeanVector(vectors);
  // std::cout << "meanVector: " << meanVector << std::endl;

  // Construct covariance matrix
  Eigen::MatrixXf covarianceMatrix = Eigen::MatrixXf::Zero(vectors[0].size(), vectors[0].size());
  // std::cout << "covarianceMatrix size: " << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << std::endl;

  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    covarianceMatrix += (vectors[i] - meanVector) * (vectors[i] - meanVector).transpose();
  }

  covarianceMatrix /= (vectors.size() - 1);

  return covarianceMatrix;
}

VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                        const unsigned int numberOfDimensions)
{
  Eigen::MatrixXf covarianceMatrix = ConstructCovarianceMatrix(vectors);
  //std::cout << "covarianceMatrix: " << covarianceMatrix << std::endl;
  std::cout << "Computed covariance matrix." << std::endl;

  typedef Eigen::JacobiSVD<Eigen::MatrixXf> SVDType;
  SVDType svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Only keep the first N singular vectors of U
  Eigen::MatrixXf truncatedU = TruncateColumns(svd.matrixU(), numberOfDimensions);


  VectorOfVectors projected;
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    projected.push_back(truncatedU.transpose() * vectors[i]);
  }

  return projected;
}

}
