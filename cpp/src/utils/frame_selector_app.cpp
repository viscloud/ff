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

#include <climits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include "saf.h"

#include "utils/frame_selector.h"
#include "utils/opencv_mp4_writer.h"

namespace po = boost::program_options;

// Set to true when the pipeline has been started. Used to signal the feeder
// thread to start, if it exists.
std::atomic<bool> started(false);

void Feeder(StreamPtr src, StreamPtr sink, int num_frames) {
  while (!started) {
    LOG(INFO) << "Waiting to start...";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  int i = 0;
  StreamReader* reader = src->Subscribe();
  while (i < num_frames) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      i++;

      if (frame->IsStopFrame()) {
        break;
      } else {
        sink->PushFrame(std::move(frame));
      }
    }
  }
  reader->UnSubscribe();

  // We need to push a stop frame in order to signal the pipeline (and
  // the rest of the application logic) to stop.
  auto stop_frame = std::make_unique<Frame>();
  stop_frame->SetValue(Frame::kFrameIdKey, ULONG_MAX);
  stop_frame->SetStopFrame(true);
  sink->PushFrame(std::move(stop_frame), true);
}

void Run(const std::string& camera_name,
         const std::string& selector_conf_filepath, int fps, int num_frames,
         const std::string& output_filepath) {
  std::vector<std::shared_ptr<Operator>> ops;

  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  ops.push_back(camera);
  auto camera_stream = camera->GetStream();
  auto camera_stream_regulated = StreamPtr(new Stream());

  auto feeder =
      std::thread([camera_stream, camera_stream_regulated, num_frames] {
        Feeder(camera_stream, camera_stream_regulated, num_frames);
      });

  auto frame_selector = std::make_shared<FrameSelector>(selector_conf_filepath);
  frame_selector->SetSource(camera_stream_regulated);
  ops.push_back(frame_selector);

  auto encoder =
      std::make_shared<OpenCvMp4Writer>("original_image", fps, output_filepath);
  encoder->SetSource(frame_selector->GetSink());
  ops.push_back(encoder);
  StreamReader* encoded_stream = encoder->GetSink()->Subscribe();

  // auto encoder = std::make_shared<GstVideoEncoder>(
  //     "original_image", output_filepath, -1, false, fps);
  // encoder->SetSource(frame_selector->GetSink());
  // ops.push_back(encoder);
  // StreamReader* encoded_stream = encoder->GetSink()->Subscribe();

  // Start the operators in reverse order.
  for (auto ops_it = ops.rbegin(); ops_it != ops.rend(); ++ops_it) {
    (*ops_it)->Start();
  }

  // Signal the feeder thread to start.
  started = true;

  while (true) {
    std::unique_ptr<Frame> frame = encoded_stream->PopFrame();
    if (frame != nullptr) {
      if (frame->IsStopFrame()) {
        LOG(INFO) << "Got stop frame";
        break;
      }
    }
  }

  // Stop the operators in forward order.
  for (const auto& op : ops) {
    op->Stop();
  }

  if (feeder.joinable()) {
    feeder.join();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Select particular frames from a video and encodes them as an H264 "
      "video.");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory which contains SAF's configuration "
                     "files");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The camera to use. This should be a file, or the program "
                     "will never stop.");
  desc.add_options()("selector-conf,s", po::value<std::string>()->required(),
                     "Path to the configuration file for the frame selector. "
                     "Text file where each line is a frame number to pick.");
  desc.add_options()("fps,f", po::value<int>()->required(),
                     "The framerate at which to encode the video.");
  desc.add_options()("num-frames,n", po::value<int>()->required(),
                     "The number of frames to read.");
  desc.add_options()("out-file,o", po::value<std::string>()->required(),
                     "The output filepath.");

  // Parse command line arguments.
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, desc), args);
    if (args.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }
    po::notify(args);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  // Set up GStreamer
  gst_init(&argc, &argv);
  // Set up glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the SAF context. This must be called before using SAF.
  Context::GetContext().Init();

  auto camera_name = args["camera"].as<std::string>();
  auto selector_conf_filepath = args["selector-conf"].as<std::string>();
  auto fps = args["fps"].as<int>();
  auto num_frames = args["num-frames"].as<int>();
  auto output_filepath = args["out-file"].as<std::string>();

  Run(camera_name, selector_conf_filepath, fps, num_frames, output_filepath);
  return 0;
}
