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

#include "ff/imagematch.h"

#include <string>
#include <thread>

#include <zmq.hpp>

constexpr auto NAME = "ImageMatch";
constexpr auto SINK_NAME = "output";
constexpr auto SOURCE_NAME = "input";

const char* ImageMatch::kMatchesKey = "ImageMatch.matches";
const char* ImageMatch::kProbsKey = "ImageMatch.match_probs";

ImageMatch::ImageMatch(std::vector<std::shared_ptr<Filter>> filters,
                       unsigned int batch_size, BatchMode batch_mode,
                       FrameLoc frame_loc)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      filters_(filters),
      batch_size_(batch_size),
      batch_mode_(batch_mode),
      frame_loc_(frame_loc),
      frames_batch_{batch_size} {}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

void ImageMatch::SetSource(StreamPtr stream) {
  Operator::SetSource(SOURCE_NAME, stream);
}

void ImageMatch::SetSink(StreamPtr stream) {
  Operator::SetSink(SINK_NAME, stream);
}

StreamPtr ImageMatch::GetSink() { return Operator::GetSink(SINK_NAME); }

std::string ImageMatch::GetName() const { return NAME; }

bool ImageMatch::Init() { return true; }

bool ImageMatch::OnStop() { return true; }

void ImageMatch::Process() {
  // Add the frame to the current batch and exit if the batch is not full.
  frames_batch_.push_back(GetFrame(SOURCE_NAME));
  if (!frames_batch_.full()) {
    return;
  }
  // The batch is full, so we will execute the microclassifiers.

  // Initialize output data structures for each frame in the batch.
  for (auto& frame : frames_batch_) {
    // Maps filter id to match result (prob > threshold).
    frame->SetValue(kMatchesKey, std::unordered_map<int, bool>());
    // Mapps filter id to match probability.
    frame->SetValue(kProbsKey, std::unordered_map<int, float>());
  }

  // Apply each microclassifier on every frame in the batch.
  for (auto filter : filters_) {
    // Outputs is populated by tensorflow::Session::Run.
    std::vector<tensorflow::Tensor> outputs;
    // Build the input Tensor from the batch's feature vectors and run it
    // through the microclassifier using TensorFlow.
    tensorflow::Status status = filter->classifier_->Run(
        {{filter->mc_input_name_, GetBatchFvs(filter)}},
        {filter->mc_output_name_}, {}, &outputs);
    if (!status.ok()) {
      LOG(FATAL) << "Session::Run() completed with errors: "
                 << status.error_message();
    }
    // MicroClassifiers should always have exactly 1 output Tensor.
    CHECK(outputs.size() == 1)
        << "Outputs should be of size 1, but is: " << outputs.size();
    // Save the results into the output data structures stored in the frame.
    // TODO: Add support for BatchMode::kStacked and both values of FrameLoc
    RecordResults(outputs.front(), filter);
  }

  // Push the batch to the next operator.
  if (batch_mode_ == kRegular) {
    for (auto& frame : frames_batch_) {
      PushFrame(SINK_NAME, std::move(frame));
    }
    // Empty the batch data structure.
    frames_batch_.clear();
  } else if (batch_mode_ == kStacked) {
    PushFrame(SINK_NAME,
              std::make_unique<Frame>(frames_batch_.at(GetTargetFrameIdx())));
  } else {
    throw std::runtime_error("Invalid BatchMode");
  }
}

int ImageMatch::GetTargetFrameIdx() const {
  if (frame_loc_ == kCurrent) {
    return frames_batch_.size() - 1;
  } else if (frame_loc_ == kMiddle) {
    return frames_batch_.size() / 2 + 1;
  } else {
    throw std::runtime_error("Invalid FrameLoc");
  }
}

tensorflow::Tensor ImageMatch::GetBatchFvs(std::shared_ptr<Filter> filter) {
  // These variables will be initialized when we process the first frame.
  auto height = 0;
  auto width = 0;
  auto channel = 0;
  // The tensor that holds the batch of feature vectors which will be passed
  // into the microclassifiers.
  tensorflow::Tensor fvs;

  // Loop over the frames in a batch. We do not extract the frame into a
  // variable because it is a unique_ptr.
  for (decltype(frames_batch_.size()) b = 0; b < frames_batch_.size(); ++b) {
    // Extract from the frame the feature vector corresponding to this
    // filter. The feature vector should have been inserted by an FvGen
    // operator, so make sure that the pipeline contains one. Clone the
    // feature vector because sometimes OpenCV throws an error if it is not
    // continuous.
    cv::Mat fv = frames_batch_.at(b)
                     ->GetValue<cv::Mat>(filter->fv_spec_.GetUniqueID())
                     .clone();

    if (b == 0) {
      // If this is the first iteration of this loop, then initialize
      // variables.
      height = fv.rows;
      width = fv.cols;
      channel = fv.channels();
      // Allocate space for the batch of feature vectors.
      tensorflow::TensorShape shape(
          {static_cast<long long>(batch_size_), height, width, channel});
      if (batch_mode_ == kStacked) {
        shape = tensorflow::TensorShape(
            {1, static_cast<long long>(batch_size_), height, width, channel});
      }
      fvs = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
    }

    // Stack Overflow suggested we do this. 4 is the number of Tensor
    // dimensions.
    if (batch_mode_ == kRegular) {
      auto fvs_mapped = fvs.tensor<float, 4>();
      // Element-wise assignment of feature vector elements into "fvs".
      for (decltype(height) i = 0; i < height; ++i) {
        for (decltype(width) j = 0; j < width; ++j) {
          for (decltype(channel) k = 0; k < channel; ++k) {
            if (batch_mode_ == kRegular) {
              fvs_mapped(b, i, j, k) =
                  fv.ptr<float>()[i * width * channel + j * channel + k];
            }
          }
        }
      }
    } else if (batch_mode_ == kStacked) {
      auto fvs_mapped = fvs.tensor<float, 5>();
      for (decltype(height) i = 0; i < height; ++i) {
        for (decltype(width) j = 0; j < width; ++j) {
          for (decltype(channel) k = 0; k < channel; ++k) {
            fvs_mapped(0, b, i, j, k) =
                fv.ptr<float>()[i * width * channel + j * channel + k];
          }
        }
      }
    } else {
      throw std::runtime_error("Invalid BatchMode");
    }
  }
  return fvs;
}

void ImageMatch::RecordResults(const tensorflow::Tensor& probs,
                               std::shared_ptr<Filter> filter) {
  int mc_num_outputs = filter->mc_num_outputs_;
  if (batch_mode_ == kRegular) {
    if (mc_num_outputs == 1) {
      auto output = probs.tensor<float, 1>();
      // Loop over the frames in a batch.
      for (decltype(frames_batch_.size()) b = 0; b < frames_batch_.size();
           ++b) {
        // Probability that frame is a match.
        float prob_match = output(b);
        // Compare probability to threshold.
        RecordResultInFrame(b, prob_match, filter);
      }
    } else if (mc_num_outputs == 2) {
      // Stack Overflow suggested we do this. 2 is the number of Tensor
      // dimensions.
      auto output = probs.tensor<float, 2>();
      // Loop over the frames in a batch.
      for (decltype(frames_batch_.size()) b = 0; b < frames_batch_.size();
           ++b) {
        // Probability that frame is a match.
        float prob_match = output(b, 0);
        // Compare probability to threshold.
        RecordResultInFrame(b, prob_match, filter);
      }
    } else {
      std::runtime_error("Invalid number of microclassifier outputs: " +
                         std::to_string(mc_num_outputs));
    }
  } else if (batch_mode_ == kStacked) {
    if (mc_num_outputs == 1) {
      // Giulio-branded stacked microclassifiers (tm) only have 1 dimension in
      // the output
      auto output = probs.tensor<float, 1>();
      float prob_match = output(0);
      RecordResultInFrame(GetTargetFrameIdx(), prob_match, filter);
    } else {
      throw std::runtime_error("Invalid number of microclassifier outputs: " +
                               std::to_string(mc_num_outputs));
    }
  } else {
    throw std::runtime_error("Invalid BatchMode");
  }
}

void LogResult(unsigned long frame_id, float prob_match, bool is_match) {
  // Log info regarding whether a frame is a match.
  LOG(INFO) << (is_match ? "Match" : "No match") << ": Frame " << frame_id
            << ": Match Confidence: " << prob_match;
}

void ImageMatch::RecordResultInFrame(int frame_idx, float prob_match,
                                     std::shared_ptr<Filter> filter) {
  bool is_match = prob_match >= filter->threshold_;
  LogResult(
      frames_batch_.at(frame_idx)->GetValue<unsigned long>(Frame::kFrameIdKey),
      prob_match, is_match);
  // Record the match information in the frame. Extract the matches data
  // structure from the frame, add the result for the current filter, then put
  // it back in the frame.
  auto matches =
      frames_batch_.at(frame_idx)->GetValue<std::unordered_map<int, bool>>(
          kMatchesKey);
  matches[filter->id_] = is_match;
  frames_batch_.at(frame_idx)->SetValue(kMatchesKey, matches);
  // Extract the probs data structure from the frame, add the result for the
  // current filter, then put it back in the frame.
  auto probs =
      frames_batch_.at(frame_idx)->GetValue<std::unordered_map<int, float>>(
          kProbsKey);
  probs[filter->id_] = prob_match;
  frames_batch_.at(frame_idx)->SetValue(kProbsKey, probs);
}
