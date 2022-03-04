// kate: replace-tabs off; indent-width 4; indent-mode normal
// vim: ts=4:sw=4:noexpandtab
/*

Copyright (c) 2010--2012,
François Pomerleau and Stephane Magnenat, ASL, ETHZ, Switzerland
You can contact the authors at <f dot pomerleau at gmail dot com> and
<stephane at magnenat dot net>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ETH-ASL BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <iostream>

#include "Eigen/SVD"

#include "ErrorMinimizersImpl.h"
#include "PointMatcherPrivate.h"
#include "Functions.h"

using namespace Eigen;
using namespace std;

typedef PointMatcherSupport::Parametrizable Parametrizable;
typedef PointMatcherSupport::Parametrizable P;
typedef Parametrizable::Parameters Parameters;
typedef Parametrizable::ParameterDoc ParameterDoc;
typedef Parametrizable::ParametersDoc ParametersDoc;

template<typename T>
PointToPlaneWithCovErrorMinimizer<T>::PointToPlaneWithCovErrorMinimizer(const Parameters& params):
	PointToPlaneErrorMinimizer<T>(PointToPlaneWithCovErrorMinimizer::availableParameters(), params),
        sensorStdDev(Parametrizable::get<T>("sensorStdDev"))
{
}

template<typename T>
typename PointMatcher<T>::TransformationParameters PointToPlaneWithCovErrorMinimizer<T>::compute(const ErrorElements& mPts_const)
{
	ErrorElements mPts = mPts_const;
    typename PointMatcher<T>::TransformationParameters out = PointToPlaneErrorMinimizer<T>::compute_in_place(mPts);

    this->covMatrix = this->estimateCovariance(mPts, out);

    return out;
}

template<typename T>
typename PointMatcher<T>::Matrix
PointToPlaneWithCovErrorMinimizer<T>::estimateCovariance(const ErrorElements& mPts, const TransformationParameters& transformation)
{
    const int max_nbr_point = mPts.reading.getNbPoints();

    Matrix bonnabel_cov(Matrix::Zero(6,6)); 
    Matrix censi_cov(Matrix::Zero(6,6));
    Matrix covariance(Matrix::Zero(6,6));
    Matrix J_hessian(Matrix::Zero(6,6));
    Matrix d2J_dReadingdX(Matrix::Zero(6, max_nbr_point));
    Matrix d2J_dReferencedX(Matrix::Zero(6, max_nbr_point));

    Vector reading_point(Vector::Zero(3));
    Vector reference_point(Vector::Zero(3));
    Vector normal(3);
    Vector reading_direction(Vector::Zero(3));
    Vector reference_direction(Vector::Zero(3));
    Vector tmp_vector_6(Vector::Zero(6));

    Matrix d2J_dZdX_bias_reading(Matrix::Zero(6, 1));
    Matrix d2J_dZdX_bias_reference(Matrix::Zero(6, 1));

    Matrix normals = mPts.reference.getDescriptorViewByName("normals");

    int valid_points_count = 0;
    for(int i = 0; i < max_nbr_point; ++i)
    {
            reading_point = mPts.reading.features.block(0,i,3,1);
            reference_point = mPts.reference.features.block(0,i,3,1);

            normal = normals.block(0,i,3,1);
      T weight = mPts.weights(0, i);
            T reading_range = reading_point.norm();
            reading_direction = reading_point / reading_range;
            T reference_range = reference_point.norm();
            reference_direction = reference_point / reference_range;

      Vector tmp_vector_3 =  transformation.block(0,0,3,3) * reference_point + transformation.block(0,3,3,1);
            T n_alpha = normal(2)*tmp_vector_3(1) - normal(1)*tmp_vector_3(2);
            T n_beta = normal(0)*tmp_vector_3(2) - normal(2)*tmp_vector_3(0);
            T n_gamma = normal(1)*tmp_vector_3(0) - normal(0)*tmp_vector_3(1);

            // update the hessian and d2J/dzdx
            tmp_vector_6 << reading_range * n_alpha, reading_range * n_beta, reading_range * n_gamma, normal(0), normal(1), normal(2);
      tmp_vector_6 *= weight;
            J_hessian += tmp_vector_6 * tmp_vector_6.transpose();

      T tmp_scalar_read = normal(0) * reading_direction(0) + normal(1) * reading_direction(1) + normal(2) * reading_direction(2);
      tmp_scalar_read = 1;
      tmp_scalar_read *= weight;
      d2J_dReadingdX.block(0,valid_points_count,6,1) = tmp_vector_6*tmp_scalar_read;
      d2J_dZdX_bias_reading += tmp_vector_6*tmp_scalar_read;

      T tmp_scalar_ref = normal(0) * reference_direction(0) + normal(1) * reference_direction(1) + normal(2) * reference_direction(2);
      tmp_scalar_ref = 1;
      tmp_scalar_ref *= weight;
            d2J_dReferencedX.block(0,valid_points_count, 6,1) = tmp_vector_6*tmp_scalar_ref;
            d2J_dZdX_bias_reference += tmp_vector_6*tmp_scalar_ref;
            valid_points_count++;
    }

    Matrix d2J_dZdX(Matrix::Zero(6, 2 * valid_points_count));
    d2J_dZdX.block(0,0,6,valid_points_count) = d2J_dReadingdX.block(0,0,6,valid_points_count);
    d2J_dZdX.block(0,valid_points_count,6,valid_points_count) = d2J_dReferencedX.block(0,0,6,valid_points_count);

    Matrix inv_J_hessian = J_hessian.inverse();

    covariance = d2J_dZdX * d2J_dZdX.transpose();
    censi_cov = inv_J_hessian * covariance * inv_J_hessian;
  bonnabel_cov = Matrix::Zero(6,6);
  bonnabel_cov += inv_J_hessian * d2J_dZdX_bias_reading * d2J_dZdX_bias_reading.transpose() * inv_J_hessian;
    bonnabel_cov += inv_J_hessian * d2J_dZdX_bias_reference * d2J_dZdX_bias_reference.transpose() * inv_J_hessian;
  bonnabel_cov += censi_cov;

  return bonnabel_cov;
}


template<typename T>
typename PointMatcher<T>::Matrix PointToPlaneWithCovErrorMinimizer<T>::getCovariance() const
{
  return covMatrix;
}

template struct PointToPlaneWithCovErrorMinimizer<float>;
template struct PointToPlaneWithCovErrorMinimizer<double>;
