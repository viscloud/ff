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

// This app profiles the CPU and memory usage of running multiple DNNs
// simultaneously.

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include "saf.h"

#include "utils/nnbench.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, const std::string& model,
         int num_copies, int batch_size, size_t queue_size, bool do_mem,
         int num_frames, const std::string& output_filepath) {
  std::vector<std::shared_ptr<Operator>> ops;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  ops.push_back(camera);

  // Create a FlowControlEntrance. Experimenting with blocking FCE.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(batch_size, true);
  fc_entrance->SetSource(camera->GetStream());
  // Experimenting with blocking FCE.
  fc_entrance->SetBlockOnPush(true);
  ops.push_back(fc_entrance);

  // Create ImageTransformer.
  ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource(fc_entrance->GetSink());
  ops.push_back(transformer);

  // Create NnBench.
  auto nnbench = std::make_shared<NnBench>(model_desc, input_shape, batch_size,
                                           num_copies);
  nnbench->SetSource(transformer->GetSink());
  ops.push_back(nnbench);

  // Create a FlowControlExit.
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(nnbench->GetSink());
  ops.push_back(fc_exit);

  // Start the operators in reverse order.
  for (auto ops_it = ops.rbegin(); ops_it != ops.rend(); ++ops_it) {
    (*ops_it)->Start(queue_size);
  }

  // Prepare output file.
  std::ofstream out_file(output_filepath);
  std::string hdr =
      "Historical fps,Num copies,Virtual mem (kb),Physical mem (kb),Total NN "
      "time";
  std::cout << hdr << std::endl;
  out_file << hdr << std::endl;

  int count = 0;
  std::string micros_key = nnbench->GetName() + ".total_micros";
  StreamReader* reader = fc_exit->GetSink("output")->Subscribe(queue_size);
  while (count < num_frames) {
    std::shared_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      ++count;
      auto historical_fps = reader->GetHistoricalFps();

      // Prepare metrics.
      // Divide by the batch size to get the time per frame.
      float nnbench_micros =
          frame->GetValue<boost::posix_time::time_duration>(micros_key)
              .total_microseconds() /
          (float)batch_size;
      int virt_mem_kb = 0;
      int phys_mem_kb = 0;
      if (do_mem) {
        virt_mem_kb = GetVirtualKB();
        phys_mem_kb = GetPhysicalKB();
      }

      // Record log line.
      std::ostringstream msg;
      msg << historical_fps << "," << num_copies << "," << virt_mem_kb << ","
          << phys_mem_kb << "," << nnbench_micros << std::endl;

      std::cout << msg.str();
      out_file << msg.str();
    }
  }
  out_file.close();

  // Terminate because of a bug in stopping the camera.
  std::terminate();

  // Stop the operators in forward order.
  // for (const auto& op : ops) {
  //   op->Stop();
  // }
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Profiles and CPU and memory usage of running multiple DNNs "
      "simultaneously.");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing SAF's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("num-copies,n", po::value<int>()->required(),
                     "THe number of copies of the model to run.");
  desc.add_options()("batch-size,b", po::value<int>()->required(),
                     "The batch size.");
  desc.add_options()("queue-size,q", po::value<size_t>()->default_value(16),
                     "The size of the queues between operators.");
  desc.add_options()("memory", "Record memory metrics too.");
  desc.add_options()("num-frames,f", po::value<int>()->required(),
                     "The number of frames to process.");
  desc.add_options()("out-file,o", po::value<std::string>()->required(),
                     "The output file.");

  // Parse the command line arguments.
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

  // Set up GStreamer.
  gst_init(&argc, &argv);
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the SAF context. This must be called before using SAF.
  Context::GetContext().Init();

  auto camera_name = args["camera"].as<std::string>();
  auto model = args["model"].as<std::string>();
  auto num_copies = args["num-copies"].as<int>();
  auto batch_size = args["batch-size"].as<int>();
  auto queue_size = args["queue-size"].as<size_t>();
  bool do_mem = args.count("memory");
  auto num_frames = args["num-frames"].as<int>();
  auto output_filepath = args["out-file"].as<std::string>();
  Run(camera_name, model, num_copies, batch_size, queue_size, do_mem,
      num_frames, output_filepath);
  return 0;
}
