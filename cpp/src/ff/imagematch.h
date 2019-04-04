// Copyright 2016 The FilterForward Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FILTERFORWARD_IMAGEMATCH_IMAGEMATCH_H_
#define FILTERFORWARD_IMAGEMATCH_IMAGEMATCH_H_

#include <memory>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <boost/circular_buffer.hpp>
#include "saf.h"

#include "ff/filter.h"

class ImageMatch : public Operator {
 public:
  // Whether a batch of frames is treated as a batch or a window for a stacked
  // MC.
  enum BatchMode { kRegular, kStacked };

  // When using a stacked MC, whether the middle frame in the window or the
  // newest frame in the window will receive the classification result.
  enum FrameLoc { kCurrent, kMiddle };

  ImageMatch(std::vector<std::shared_ptr<Filter>> filters,
             unsigned int batch_size = 1, BatchMode batch_mode = kRegular,
             FrameLoc frame_loc = kMiddle);

  static std::shared_ptr<ImageMatch> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Operator::SetSource;

  void SetSink(StreamPtr stream);
  using Operator::SetSink;

  StreamPtr GetSink();
  using Operator::GetSink;

  std::string GetName() const override;

  static const char* kMatchesKey;
  static const char* kProbsKey;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // Returns a float Tensor containing the specified query's feature vectors
  // for all of the frames in the current batch. This function extracts feature
  // vectors from every frame in the batch and then performs an element-wise
  // copy into the output Tensor. The feature vectors stored in the frames must
  // contain floats. The return shape is automatically inferred from the feature
  // vectors stored. The specific feature vector to use is defined in the
  // query's FvSpec (see src/operator/FvGen.h).
  tensorflow::Tensor GetBatchFvs(std::shared_ptr<Filter> filter);
  // Parses the results Tensor produced by running the specified filter and
  // populates the current batch's data structures with the filter's match
  // results. The results Tensor's dimensions are [batch size x 2], where the 2
  // is [probability not a match, probability a match]. This function also logs
  // a summary of each frame's match results.
  void RecordResults(const tensorflow::Tensor& probs,
                     std::shared_ptr<Filter> filter);
  void RecordResultInFrame(int frame_idx, float prob_match,
                           std::shared_ptr<Filter> filter);
  int GetTargetFrameIdx() const;
  std::vector<std::shared_ptr<Filter>> filters_;
  // Batch size should be above 0
  // batch_size_ determines how many frames should be run through the MC at once
  // using the batching feature of tensorflow
  unsigned int batch_size_;
  BatchMode batch_mode_;
  FrameLoc frame_loc_;
  // cur_batch_frames holds the actual frames in the batch
  boost::circular_buffer<std::unique_ptr<Frame>> frames_batch_;
};

#endif  // FILTERFORWARD_IMAGEMATCH_IMAGEMATCH_H_
