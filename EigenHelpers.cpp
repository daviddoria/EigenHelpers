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

Eigen::VectorXf STDVectorToEigenVector(const std::vector<float>& vec)
{
  Eigen::VectorXf eigenVector(vec.size());
  for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); ++i)
  {
    eigenVector[i] = vec[i];
  }
  // Could alternatively use an Eigen Map: http://eigen.tuxfamily.org/dox-devel/TutorialMapClass.html
  return eigenVector;
}

std::vector<float> EigenVectorToSTDVector(const Eigen::VectorXf& vec)
{
  std::vector<float> stdvector(vec.size());
  for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); ++i)
  {
    stdvector[i] = vec[i];
  }
  // Could equivalently do:
  // std::vector<float> stdvector(vec.data(), vec.data()+vec.size());
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
  unsigned int numberOfVectors = vectors.size();

  if(numberOfVectors == 0)
  {
    throw std::runtime_error("Can't compute the covariance matrix of a list of vectors of length zero!");
  }

  unsigned int numberOfDimensions = vectors[0].size();

  Eigen::VectorXf meanVector = ComputeMeanVector(vectors);
  for(unsigned int i = 0; i < meanVector.size(); ++i)
  {
    if(meanVector[i] != meanVector[i]) // check for NaN
    {
      throw std::runtime_error("meanVector cannot contain NaN!");
    }
  }
  // std::cout << "meanVector: " << meanVector << std::endl;

  // Construct covariance matrix
  Eigen::MatrixXf covarianceMatrix = Eigen::MatrixXf::Zero(numberOfDimensions, numberOfDimensions);
  // std::cout << "covarianceMatrix size: " << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << std::endl;

  for(unsigned int i = 0; i < numberOfVectors; ++i)
  {
    covarianceMatrix += (vectors[i] - meanVector) * (vectors[i] - meanVector).transpose();
  }

  covarianceMatrix /= (numberOfVectors - 1); // this is the "N-1" for an unbiased estimate

  return covarianceMatrix;
}

EigenHelpers::VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                                      const Eigen::MatrixXf& covarianceMatrix,
                                                      const unsigned int numberOfDimensions)
{
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

EigenHelpers::VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                                      const Eigen::MatrixXf& covarianceMatrix,
                                                      const float singularValueWeightToKeep)
{
  // Compute the SVD
  typedef Eigen::JacobiSVD<Eigen::MatrixXf> SVDType;
  SVDType svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

  //std::cout << "There are " << svd.singularValues().size() << " singular values." << std::endl;

  // Compute the sum of the singular values
  float singularValueSum = 0.0f;
  SVDType::SingularValuesType singularValues = svd.singularValues();
  std::cout << "SingularValues: ";
  for(SVDType::Index i = 0; i < singularValues.size(); ++i)
  {
    singularValueSum += singularValues[i];
    std::cout << singularValues[i] << " ";
  }

  std::cout << std::endl;
  // Determine how many vectors we need to keep to keep the desired amount of "eigen weight"

  float normalizedSingularVectorSum = 0.0f;
  unsigned int numberOfDimensions = 0;
  for(SVDType::Index i = 0; i < singularValues.size(); ++i)
  {
    numberOfDimensions++;
    normalizedSingularVectorSum += singularValues[i]/singularValueSum;
    if(normalizedSingularVectorSum > singularValueWeightToKeep)
    {
      break;
    }
  }

  // Only keep the first N singular vectors of U
  Eigen::MatrixXf truncatedU = TruncateColumns(svd.matrixU(), numberOfDimensions);

  VectorOfVectors projected;
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    projected.push_back(truncatedU.transpose() * vectors[i]);
  }

  return projected;
}

VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                        const unsigned int numberOfDimensions)
{
  Eigen::MatrixXf covarianceMatrix = ConstructCovarianceMatrix(vectors);
  std::cout << "covarianceMatrix: " << covarianceMatrix << std::endl;

  return DimensionalityReduction(vectors, covarianceMatrix, numberOfDimensions);
}

void Standardize(EigenHelpers::VectorOfVectors& vectors)
{
  // Subtract the mean and divide by the standard devation of each set of corresponding elements.

  // Each component of the result is the mean of the corresponding collection of elements. That is,
  // the 0th component of 'meanVector' is the mean of all of the 0th components in 'vectors'.
  Eigen::VectorXf meanVector = ComputeMeanVector(vectors);

  // Variance = 1/NumPixels * sum_i (x_i - u)^2
  Eigen::VectorXf standardDeviationVector(vectors[0].size());

  // Loop over each element
  for(Eigen::VectorXf::Index element = 0; element < vectors[0].size(); ++element)
  {
    float sumOfDifferenceFromMean = 0.0f;
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      sumOfDifferenceFromMean += pow(vectors[i][element] - meanVector[element], 2.0f);
    }
    standardDeviationVector[element] = sqrt(sumOfDifferenceFromMean / static_cast<float>(vectors.size() - 1));
  }

  // Actually subtract the mean and divide by the standard deviation
  for(Eigen::VectorXf::Index element = 0; element < vectors[0].size(); ++element)
  {
    for(size_t i = 0; i < vectors.size(); ++i)
    {
      vectors[i][element] -= meanVector[element];
      vectors[i][element] /= standardDeviationVector[element];
    }
  }
}

} // end EigenHelpers namespace
