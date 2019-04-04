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

#include <unistd.h>

#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "saf.h"

#include "ff/deduplicator.h"
#include "ff/ed.h"
#include "ff/imagematch.h"
#include "fv_gen.h"

namespace po = boost::program_options;

constexpr auto JPEG_WRITER_FIELD = "original_image";
std::unordered_set<std::string> TRANSMIT_FIELDS(
    {Camera::kCaptureTimeMicrosKey, Ed::kEventIdKey, Frame::kFrameIdKey,
     ImageMatch::kMatchesKey, ImageMatch::kProbsKey});

// Used to signal all threads that the pipeline should stop.
std::atomic<bool> stopped(false);

typedef struct {
  int num;
  std::string layer;
  int xmin;
  int ymin;
  int xmax;
  int ymax;
  bool flat;
  std::string path;
  float threshold;
} query_spec_t;

// Designed to be run in its own thread. Sets "stopped" to true after num_frames
// have been processed or after a stop frame has been detected.
void Stopper(StreamPtr stream, unsigned int num_frames) {
  unsigned int count = 0;
  StreamReader* reader = stream->Subscribe();
  while (!stopped && (num_frames == 0 || count < num_frames)) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    ++count;
    if (frame != nullptr && frame->IsStopFrame()) {
      break;
    } else {
      LOG(INFO) << "FF received frame: "
                << frame->GetValue<unsigned long>(Frame::kFrameIdKey) << " ("
                << count << " total)";
    }
  }
  stopped = true;
  reader->UnSubscribe();
}

// Designed to be run in its own thread. Creates a log file containing
// performance metrics for the specified stream. If there are multiple Logger
// threads per instance of FilterForward, then the log filepath needs to be
// unique.
void Logger(StreamPtr stream, std::vector<std::string> network_fields,
            const std::string& log_filepath, bool log_memory, bool force_kill) {
  bool is_first_frame = true;
  // Total bytes sent over the network.
  double total_bytes = 0;
  // The time at which we received the first frame.
  boost::posix_time::ptime start_time;
  // Used for tracking how often we output the log messages.
  boost::posix_time::ptime previous_time;
  // The contents of this will be written to the log file when the pipeline
  // stops.
  std::ostringstream log;

  // Loop until the stopper thread signals that we need to stop.
  StreamReader* reader = stream->Subscribe();
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame(100);
    if (frame == nullptr) {
      continue;
    } else if (frame->IsStopFrame()) {
      // We still need to check for stop frames, even though the stopper thread
      // is watching for stop frames at the highest level.
      break;
    } else {
      if (is_first_frame) {
        // We start recording after the first frame in order to reduce
        // experimental error when calculating the network bandwidth usage.
        is_first_frame = false;
        start_time = boost::posix_time::microsec_clock::local_time();
        previous_time = start_time;
      } else {
        // Extract all of the performance metrics. Use total_microseconds() to
        // avoid dividing by zero if less than a second has passed.
        double fps = reader->GetHistoricalFps();
        boost::posix_time::ptime current_time =
            boost::posix_time::microsec_clock::local_time();
        long latency_micros =
            (current_time - frame->GetValue<boost::posix_time::ptime>(
                                Camera::kCaptureTimeMicrosKey))
                .total_microseconds();
        long it_micros = frame
                             ->GetValue<boost::posix_time::time_duration>(
                                 "ImageTransformer.total_micros")
                             .total_microseconds();
        long nne_micros = frame
                              ->GetValue<boost::posix_time::time_duration>(
                                  "NeuralNetEvaluator.total_micros")
                              .total_microseconds();
        long fvg_micros = frame
                              ->GetValue<boost::posix_time::time_duration>(
                                  "FvGen.total_micros")
                              .total_microseconds();
        long im_micros = frame
                             ->GetValue<boost::posix_time::time_duration>(
                                 "ImageMatch.total_micros")
                             .total_microseconds();
        long kv_micros =
            frame->GetValue<boost::posix_time::time_duration>(Ed::kKvMicrosKey)
                .total_microseconds();
        long ed_micros =
            frame->GetValue<boost::posix_time::time_duration>("Ed.total_micros")
                .total_microseconds();

        int physical_kb = 0;
        int virtual_kb = 0;
        if (log_memory) {
          physical_kb = GetPhysicalKB();
          virtual_kb = GetVirtualKB();
        }
        // Calculate the network bandwidth.
        total_bytes += frame->GetRawSizeBytes(std::unordered_set<std::string>{
            network_fields.begin(), network_fields.end()});
        double net_bw_bps = (total_bytes * 8 /
                             (current_time - start_time).total_microseconds()) *
                            1000000;

        // Assemble log message;
        std::ostringstream msg;
        msg << fps << "," << latency_micros << "," << it_micros << ","
            << nne_micros << "," << fvg_micros << "," << im_micros << ","
            << kv_micros << "," << ed_micros << "," << physical_kb << ","
            << virtual_kb << "," << net_bw_bps;
        if ((current_time - previous_time).total_seconds() >= 1) {
          // Every one second, log a frame's metrics to the console.
          std::cout << msg.str() << std::endl;
          previous_time = current_time;
        }
        // Always log to the log file.
        log << msg.str() << std::endl;
      }
    }
  }
  reader->UnSubscribe();

  std::ofstream log_file(log_filepath);
  log_file << "# historical fps, end-to-end latency (micros), "
              "ImageTransformer micros, NNE micros, FvGen micros, "
              "ImageMatch micros, k-voting micros, "
              "Ed micros, physical memory (kb), virtual memory (kb), "
              "network bandwidth (bps)"
           << std::endl
           << log.str();
  log_file.close();

  if (force_kill) {
    LOG(INFO) << "Terminating...";
    std::terminate();
  }
}

// Posts ImageMatch matches from the provided stream to Slack. Constrains post
// rate to once every "post_delta_ms" milliseconds.
void Slack(StreamPtr stream, const std::string& slack_url,
           const std::string& output_dir,
           boost::posix_time::time_duration post_delta_ms) {
  CURL* curl;
  CURLcode res;
  curl_global_init(CURL_GLOBAL_ALL);

  StreamReader* reader = stream->Subscribe();
  // The time of the last Slack post.
  boost::posix_time::ptime last_post_ms;
  unsigned long last_match_id = 0;
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame == nullptr) {
      continue;
    } else if (frame->IsStopFrame()) {
      // We still need to check for stop frames, even though the stopper thread
      // is watching for stop frames at the highest level.
      break;
    } else {
      if (frame->Count(ImageMatch::kMatchesKey) &&
          frame
              ->GetValue<std::unordered_map<int, bool>>(ImageMatch::kMatchesKey)
              .begin()
              ->second) {
        // Calculate the number of frames since the last match.
        unsigned long frame_id = frame->GetValue<unsigned long>("frame_id");
        unsigned long delta_frames = frame_id - last_match_id;
        last_match_id = frame_id;

        // Post if either this is the first match or sufficient time has passed
        // since the last post.
        boost::posix_time::ptime time_ms =
            boost::posix_time::microsec_clock::local_time();
        if (last_post_ms.is_not_a_date_time() ||
            (time_ms - last_post_ms) >= post_delta_ms) {
          last_post_ms = time_ms;

          // Construct the message.
          std::string time =
              GetDateTimeString(frame->GetValue<boost::posix_time::ptime>(
                  Camera::kCaptureTimeMicrosKey));
          float match_prob = frame->GetValue<std::unordered_map<int, float>>(
              ImageMatch::kProbsKey)[0];
          // Extract the relative path to the frame JPEG.
          std::string filepath =
              frame->GetValue<std::string>(JpegWriter::kPathKey);
          filepath.erase(0, output_dir.size() + 1);
          std::string frame_link =
              "http://istc-vcs.pc.cc.cmu.edu:8001/" + filepath;

          std::vector<std::string> tokens = SplitString(filepath, "/");
          CHECK(tokens.size() == 3) << "Malformed frame filepath!";
          std::string day = tokens.at(0);
          std::string hour = tokens.at(1);
          std::string file = tokens.at(2);
          std::string thumbnail_link =
              "http://istc-vcs.pc.cc.cmu.edu:8001/thumb?day=" + day +
              "&hour=" + hour + "&file=" + file;

          std::string msg =
              "{\"text\":\"Detection at " + time +
              " (delta: " + std::to_string(delta_frames) +
              " frames), match confidence: " + std::to_string(match_prob) +
              "\nFull image: <" + frame_link + "|" +
              frame_link.substr(7, frame_link.size() - 7) + ">\n<" +
              thumbnail_link + "|Thumbnail>\"\n}";

          // Send the message.
          curl = curl_easy_init();
          if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, slack_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, msg.c_str());
            res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
              LOG(WARNING) << "Curl failed: " << curl_easy_strerror(res);
            }
          } else {
            LOG(WARNING) << "Curl init failed.";
          }
        }
      }
    }
  }

  curl_global_cleanup();
  reader->UnSubscribe();
}

void Run(const std::string& ff_conf, unsigned int num_frames, bool block,
         size_t queue_size, bool use_camera, const std::string& camera_name,
         const std::string& publish_url, unsigned int file_fps,
         int throttled_fps, unsigned int tokens, const std::string& model,
         size_t nne_batch_size, bool mc_stacked, int mc_batch_size,
         int kv_window, bool kv_whole_buffer, bool mc_pass_all,
         std::vector<std::string> fields, const std::string& log_filepath,
         const std::string& output_dir, bool save_video, bool save_matches,
         bool log_memory, bool slack, const std::string& slack_url,
         boost::posix_time::time_duration post_delta_ms, int angle,
         bool force_kill, bool crop) {
  boost::filesystem::path output_dir_path(output_dir);
  boost::filesystem::create_directory(output_dir_path);

  // Parse the ff_conf file.
  std::vector<std::shared_ptr<Filter>> filters;
  std::ifstream ff_conf_file(ff_conf);
  std::string line;
  while (std::getline(ff_conf_file, line)) {
    std::vector<std::string> args = SplitString(line, ",");
    if (StartsWith(line, "#")) {
      // Ignore comment lines.
      continue;
    }
    CHECK(args.size() == 14) << "Malformed configuration file.";

    for (int i = 0; i < std::stoi(args.at(0)); ++i) {
      filters.push_back(std::make_shared<Filter>(
          filters.size(),
          FvSpec(args.at(1), std::stoi(args.at(2)), std::stoi(args.at(3)),
                 std::stoi(args.at(4)), std::stoi(args.at(5))),
          args.at(6), args.at(7), args.at(8), std::stoi(args.at(9)),
          std::stof(args.at(10)), std::stoi(args.at(11)),
          std::stoul(args.at(12)), std::stol(args.at(13))));
    }
  }

  std::vector<std::shared_ptr<Operator>> ops;

  StreamPtr input_stream;

  if (use_camera) {
    // Create a Camera.
    std::shared_ptr<Camera> camera =
        CameraManager::GetInstance().GetCamera(camera_name);
    // Why is this false? Does the camera always need to be able to push
    // frames to prevent unbounded memory use growth?
    camera->SetBlockOnPush(false);
    ops.push_back(camera);
    input_stream = camera->GetStream();

    if (camera->GetCameraType() == CameraType::CAMERA_TYPE_GST) {
      std::shared_ptr<GSTCamera> gst_camera =
          std::dynamic_pointer_cast<GSTCamera>(camera);
      // gst_camera->SetFileFramerate(file_fps);
      if (save_video) {
        gst_camera->SetOutputFilepath(output_dir + "/" + camera_name + ".mp4");
      }
    } else {
      LOG(WARNING)
          << "\"--file-fps=" << file_fps
          << "\" is only supported for GStreamer cameras (that read files)!";
      if (save_video) {
        LOG(WARNING)
            << "\"--save-video\" is only supported for GStreamer cameras!";
      }
    }
  } else {
    // Create a FrameSubscriber.
    auto subscriber = std::make_shared<FrameSubscriber>(publish_url);
    // This is false because it is false in the "use_camera" case.
    subscriber->SetBlockOnPush(false);
    ops.push_back(subscriber);
    input_stream = subscriber->GetSink();
  }

  StreamPtr correct_fps_stream = input_stream;
  if (throttled_fps > 0) {
    // If we are supposed to throttler the stream in software, then create a
    // Throttler.
    auto throttler = std::make_shared<Throttler>(throttled_fps);
    throttler->SetSource(input_stream);
    throttler->SetBlockOnPush(false);
    ops.push_back(throttler);
    correct_fps_stream = throttler->GetSink();
  }

  // Create a FlowControlEntrance. Experimenting with blocking FCE.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(tokens, true);
  fc_entrance->SetSource(correct_fps_stream);
  // Experimenting with blocking FCE.
  fc_entrance->SetBlockOnPush(true);
  ops.push_back(fc_entrance);

  // Create an ImageTransformer.
  ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, crop, angle);
  transformer->SetSource(fc_entrance->GetSink());
  transformer->SetBlockOnPush(block);
  ops.push_back(transformer);

  // Create a NeuralNetEvaluator.
  auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape,
                                                  nne_batch_size);
  nne->SetSource(transformer->GetSink());
  nne->SetBlockOnPush(block);
  ops.push_back(nne);

  // Create an FvGen.
  auto fvgen = std::make_shared<FvGen>();
  fvgen->SetSource(nne->GetSink());
  fvgen->SetBlockOnPush(block);
  ops.push_back(fvgen);

  // Register the necessary spatial crops for each filter.
  for (auto filter : filters) {
    nne->PublishLayer(filter->fv_spec_.layer_name_);
    fvgen->AddFv(filter->fv_spec_);
  }

  // Create an ImageMatch.
  ImageMatch::BatchMode batch_mode;
  if (mc_stacked) {
    batch_mode = ImageMatch::BatchMode::kStacked;
  } else {
    batch_mode = ImageMatch::BatchMode::kRegular;
  }
  auto im = std::make_shared<ImageMatch>(filters, mc_batch_size, batch_mode,
                                         ImageMatch::FrameLoc::kMiddle);
  im->SetSource(fvgen->GetSink());
  im->SetBlockOnPush(block);
  ops.push_back(im);

  // Create a FlowControlExit.
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(im->GetSink());
  fc_exit->SetBlockOnPush(block);
  ops.push_back(fc_exit);

  // Create an EventDetector.
  auto ed =
      std::make_shared<Ed>(filters, kv_window, kv_whole_buffer, mc_pass_all);
  ed->SetSource(fc_exit->GetSink());
  ed->SetBlockOnPush(block);
  ops.push_back(ed);

  // Assemble final frame fields.
  for (const auto& op : ops) {
    fields.push_back(op->GetName() + ".total_micros");
  }
  fields.insert(fields.end(), TRANSMIT_FIELDS.begin(), TRANSMIT_FIELDS.end());

  // Create a Deduplicator.
  auto dedup = std::make_shared<Deduplicator>(
      std::unordered_set<std::string>(fields.begin(), fields.end()));
  dedup->SetSource(ed->GetSink());
  dedup->SetBlockOnPush(block);
  ops.push_back(dedup);
  StreamPtr dedup_stream = dedup->GetSink();

  if (save_matches || slack) {
    // Create JpegWriter.
    auto jpeg_writer =
        std::make_shared<JpegWriter>(JPEG_WRITER_FIELD, output_dir, true);
    jpeg_writer->SetSource(dedup_stream);
    ops.push_back(jpeg_writer);
    dedup_stream = jpeg_writer->GetSink();
  }
  if (save_matches) {
    // Create FrameWriter.
    auto frame_writer = std::make_shared<FrameWriter>(
        TRANSMIT_FIELDS, output_dir, FrameWriter::FileFormat::JSON, false,
        true);
    frame_writer->SetSource(dedup_stream);
    ops.push_back(frame_writer);
    dedup_stream = frame_writer->GetSink();
  }

  // Launch stopper thread.
  std::thread stopper_thread(
      [dedup_stream, num_frames] { Stopper(dedup_stream, num_frames); });
  // Launch logger thread.
  std::thread logger_thread(
      [dedup_stream, fields, log_filepath, log_memory, force_kill] {
        Logger(dedup_stream, fields, log_filepath, log_memory, force_kill);
      });
  // Launch Slack thread
  std::thread slack_thread;
  if (slack) {
    slack_thread =
        std::thread([dedup_stream, slack_url, output_dir, post_delta_ms] {
          Slack(dedup_stream, slack_url, output_dir, post_delta_ms);
        });
  }

  // Start the operators in reverse order.
  for (auto ops_it = ops.rbegin(); ops_it != ops.rend(); ++ops_it) {
    (*ops_it)->Start(queue_size);
  }

  if (num_frames > 0) {
    while (!stopped) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } else {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
    stopped = true;
  }

  // Stop the operators in forward order.
  for (const auto& op : ops) {
    op->Stop();
  }

  // Join all of our helper threads.
  stopper_thread.join();
  logger_thread.join();
  if (slack_thread.joinable()) {
    slack_thread.join();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("FilterForward");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>()->required(),
                     "The directory containing SAF's configuration "
                     "files.");
  desc.add_options()("ff-conf,f", po::value<std::string>(),
                     "The file containing the FilterForward's "
                     "configuration.");
  desc.add_options()("num-frames", po::value<unsigned int>()->default_value(0),
                     "The number of frames to run before automatically "
                     "stopping.");
  desc.add_options()("block,b",
                     "Whether operators should block when pushing frames.");
  desc.add_options()("queue-size,q", po::value<size_t>()->default_value(16),
                     "The size of the queues between operators.");
  desc.add_options()("camera,c", po::value<std::string>(),
                     "The name of the camera to use. Overrides "
                     "\"--publish-url\".");
  desc.add_options()("publish-url,u", po::value<std::string>(),
                     "The URL (host:port) on which the frame stream is being "
                     "published.");
  desc.add_options()("file-fps", po::value<unsigned int>()->default_value(0),
                     "Rate at which to read a file source (no effect if not "
                     "file source).");
  desc.add_options()("throttled-fps,i", po::value<int>()->default_value(0),
                     "The FPS at which to throttle (in software) the camera "
                     "stream. 0 means no throttling.");
  desc.add_options()("tokens,t", po::value<unsigned int>()->default_value(5),
                     "The number of flow control tokens to issue.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("nne-batch-size,s", po::value<size_t>()->default_value(1),
                     "nne batch size");
  desc.add_options()("mc-stacked",
                     "Whether the microclassifiers use a stack of frames");
  desc.add_options()("mc-batch-size", po::value<int>()->default_value(5),
                     "Microclassifier batch/stack size.");
  desc.add_options()("kv-window", po::value<int>()->default_value(11),
                     "K-Voting window size.");
  desc.add_options()("kv-whole-buffer",
                     "Whether to apply k-voting to a whole buffer or each "
                     "frame individually.");
  desc.add_options()("mc-pass-all",
                     "Whether to ignore the results of the MC and k-voting and "
                     "push every frame.");
  desc.add_options()("fields",
                     po::value<std::vector<std::string>>()
                         ->multitoken()
                         ->composing()
                         ->required(),
                     "The fields to send over the network when calculating "
                     "theoretical network bandwidth usage.");
  desc.add_options()("log-file,l", po::value<std::string>()->required(),
                     "The path at which the log file should be stored.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to write output data.");
  desc.add_options()("save-video",
                     "Whether to save the original video stream to disk.");
  desc.add_options()(
      "save-matches",
      "Save JPEGs of frames matched by the first level of the hierarchy.");
  desc.add_options()("memory-usage", "Log memory usage.");
  desc.add_options()("slack", po::value<std::string>(),
                     "Enable Slack notifications for matched frames, and send "
                     "notifications to the provided hook url.");
  desc.add_options()("post-delta-ms", po::value<long>()->default_value(0),
                     "The minimum interval between Slack posts.");
  desc.add_options()("rotate,r", po::value<int>(),
                     "The angle to rotate frames; must be 0, 90, 180, or 270.");
  desc.add_options()(
      "force-kill",
      "Whether to forcibly kill the process after saving the log file. This is "
      "useful to prefend nondeterministic hanging in test scripts.");
  desc.add_options()(
      "crop",
      "If the incoming frame dimensions are not the same as the base DNN's "
      "input size, then crop (fast) instead of resizing (slow).");

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

  Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  // Initialize the SAF context. This must be called before using SAF.
  Context::GetContext().Init();

  std::string ff_conf = args["ff-conf"].as<std::string>();
  unsigned int num_frames = args["num-frames"].as<unsigned int>();
  bool block = args.count("block");
  size_t queue_size = args["queue-size"].as<size_t>();
  std::string camera;
  bool use_camera = args.count("camera");
  if (use_camera) {
    camera = args["camera"].as<std::string>();
  }
  std::string publish_url;
  if (args.count("publish-url")) {
    publish_url = args["publish-url"].as<std::string>();
  } else if (!use_camera) {
    throw std::runtime_error(
        "Must specify either \"--camera\" or \"--publish-url\".");
  }
  unsigned int file_fps = args["file-fps"].as<unsigned int>();
  int throttled_fps = args["throttled-fps"].as<int>();
  unsigned int tokens = args["tokens"].as<unsigned int>();
  std::string model = args["model"].as<std::string>();
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  bool mc_stacked = args.count("mc-stacked");
  int mc_batch_size = args["mc-batch-size"].as<int>();
  int kv_window = args["kv-window"].as<int>();
  bool kv_whole_buffer = args.count("kv-whole-buffer");
  bool mc_pass_all = args.count("mc-pass-all");
  std::vector<std::string> fields =
      args["fields"].as<std::vector<std::string>>();
  std::string log_filepath = args["log-file"].as<std::string>();
  std::string output_dir = args["output-dir"].as<std::string>();
  bool save_video = args.count("save-video");
  bool save_matches = args.count("save-matches");
  bool log_memory = args.count("memory-usage");
  bool slack = args.count("slack");
  int angle = 0;
  if (args.count("rotate")) {
    auto angles = std::set<int>{0, 90, 180, 270};
    int angle = args["rotate"].as<int>();
    if (angles.find(angle) == angles.end()) {
      std::ostringstream msg;
      msg << "\"--rotate\" angle must be 0, 90, 180, or 270, but is: " << angle;
      throw std::runtime_error(msg.str());
    }
  }
  std::string slack_url;
  if (slack) {
    slack_url = args["slack"].as<std::string>();
  }
  long post_delta_ms_l = args["post-delta-ms"].as<long>();
  if (post_delta_ms_l < 0) {
    std::ostringstream msg;
    msg << "Value for \"--post-delta-ms\" cannot be negative, but is: "
        << post_delta_ms_l;
    throw std::runtime_error(msg.str());
  }
  boost::posix_time::time_duration post_delta_ms =
      boost::posix_time::milliseconds(post_delta_ms_l);
  bool force_kill = args.count("force-kill");
  bool crop = args.count("crop");

  Run(ff_conf, num_frames, block, queue_size, use_camera, camera, publish_url,
      file_fps, throttled_fps, tokens, model, nne_batch_size, mc_stacked,
      mc_batch_size, kv_window, kv_whole_buffer, mc_pass_all, fields,
      log_filepath, output_dir, save_video, save_matches, log_memory, slack,
      slack_url, post_delta_ms, angle, force_kill, crop);
  return 0;
}
