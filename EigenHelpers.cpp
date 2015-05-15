#include "EigenHelpers.h"

namespace EigenHelpers
{

void GetBoundingBox(const Eigen::MatrixXd& data, Eigen::VectorXd& minCorner, Eigen::VectorXd& maxCorner)
{
  assert(data.cols() > 0);

  minCorner.resize(data.rows());
  maxCorner.resize(data.rows());

  for(unsigned int coordinate = 0; coordinate < data.rows(); ++coordinate)
  {
    minCorner[coordinate] = std::numeric_limits<double>::max();
    maxCorner[coordinate] = std::numeric_limits<double>::min();

    for(unsigned int pointId = 0; pointId < data.cols(); ++pointId)
    {
      if(data(coordinate, pointId) > maxCorner(coordinate))
      {
        maxCorner(coordinate) = data(coordinate, pointId);
      }

      if(data(coordinate, pointId) < minCorner(coordinate))
      {
        minCorner(coordinate) = data(coordinate, pointId);
      }
    }
  }
}

} // end EigenHelpers namespace
