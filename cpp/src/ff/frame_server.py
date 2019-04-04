#!/usr/bin/python3

# Copyright 2016 The FilterForward Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from io import BytesIO
import json
import os
from os import path

from cachetools import LRUCache
import cherrypy
from PIL import Image


INDEX_TEMPATE = """
<head><title>FilterForward Frame Server</title></head>
<body>
{}
</body>
"""
IMAGES_STYLE = """<style>
  .cards {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    flex-direction: row;
    max-height: 100vh;
  }
  .cards p {
    text-align: center;
    margin: 2px;
    border: 1px solid #000;
    //box-shadow: 3px 3px 8px 0px rgba(0,0,0,0.3);
    max-width: 23vw;
  }
</style>
<div class="cards">
"""


def get_dir_list(dirpath):
    dirlist = os.listdir(dirpath)
    try:
        dirlist = sorted(dirlist, key=int)
    except ValueError:
        dirlist = sorted(dirlist)
    return dirlist


def show_gallery(dirpath, thumbnail_size):
    dirlist = get_dir_list(dirpath)
    json_paths = [f for f in dirlist if f.endswith(".json")]
    # Only pick one in 10 to prevent visual overload.
    json_paths = json_paths[::10]

    out = IMAGES_STYLE

    for json_path in json_paths:
        no_extension = path.splitext(json_path)[0]
        with open(path.join(dirpath, json_path)) as img_file:
            j = json.loads(img_file.read())

        # Extract metadata. These need to be guarded in case a value is not
        # available.
        if "capture_time_micros" in j:
            date = j["capture_time_micros"]
        else:
            date = "Unknown date"
        if "frame_id" in j:
            frame_id = j["frame_id"]
        else:
            frame_id = "Unknown"
        if "ImageMatch.match_probs" in j:
            # For the train pipeline, we assume that there is only one query in
            # FilterForward.
            prob = "{:.3f}".format(float(j["ImageMatch.match_probs"]["0"]))
        else:
            prob = "Unknown"

        img_path = f"{no_extension}_original_image.jpg"
        thumb_path = f"/thumb{cherrypy.request.path_info}{img_path}"
        img_container = f"""
            <p>
                <a href="{img_path}">
                    <img height={thumbnail_size}px width={thumbnail_size}px src="{thumb_path}" />
                </a>
                <br />{date}
                <br />Frame ID: {frame_id}
                <br />Match Prob: {prob}
            </p>
        """
        out += img_container

    return out


@cherrypy.popargs("day", "hour")
class Root(object):

    def __init__(self, root_dir, thumbnail_size, angle):
        self.root_dir = root_dir
        self.thumbnail_size = thumbnail_size
        self.angle = angle
        self.cache = LRUCache(maxsize=5000)

    @cherrypy.expose
    def index(self, day=None, hour=None):
        body = ""

        # Show thumbnail gallery for given hour
        if day is not None and hour is not None:
            dirpath = f"{self.root_dir}/{day}/{hour}"
            res = show_gallery(dirpath, self.thumbnail_size)
            body += res
            return INDEX_TEMPATE.format(body)

        # Show list of directories
        if day is not None:
            dirpath = f"{self.root_dir}/{day}"
        else:
            dirpath = f"{self.root_dir}"
        res = get_dir_list(dirpath)
        body += f"<p><b>Current Path:</b> {dirpath}</p>"
        body += " ".join([f"<li><a href=\"{i}\">{i}</a></li>" for i in res])
        return INDEX_TEMPATE.format(body)

    @cherrypy.expose
    def thumb(self, day=None, hour=None, file=None):
        filepath = f"{self.root_dir}/{day}/{hour}/{file}"
        if filepath in self.cache:
            print(f"Found cached thumbnail for: {filepath}")
            bio = BytesIO(self.cache[filepath])
        else:
            try:
                with Image.open(filepath) as img:
                    img.thumbnail((self.thumbnail_size, self.thumbnail_size))
                    img_rotated = img.rotate(self.angle, expand=True)
                    bio = BytesIO()
                    # 95 is the highest quality that actually yields appreciable
                    # compression.
                    img_rotated.save(bio, format="JPEG", quality=95)

                    print(f"Caching thumbnail for: {filepath}")
                    self.cache[filepath] = bio.getvalue()
            except IOError:
                print("IO error when generating thumbnail for: {}".format(
                    filepath))
                return

        cherrypy.response.headers["Content-type"] = "image/jpg"
        bio.seek(0)
        return cherrypy.lib.file_generator(bio)


def main():
    parser = argparse.ArgumentParser(
        description=("Frame server for a temporally-organized frames hierarchy "
                     "created by SAF"))
    parser.add_argument(
        "--dir",
        "-d",
        dest="root_dir",
        help="The location of the frames directory hierarchy.",
        required=True)
    parser.add_argument(
        "--size",
        "-s",
        dest="thumbnail_size",
        default=200,
        help="The size of the thumbnail images (largest dimension).",
        type=int)
    parser.add_argument(
        "--port",
        "-p",
        default=8000,
        help="The port on which to listen for request.",
        type=int)
    parser.add_argument(
        "--rotate",
        "-r",
        default=0,
        help="The angle by which to rotate the images and thumbnails.",
        type=int)
    args = parser.parse_args()

    root_dir = args.root_dir
    conf = {
        "global": {
            "server.socket_host": "0.0.0.0",
            "server.socket_port": args.port
        },
        "/": {
            "tools.staticdir.root": path.abspath(root_dir),
            "tools.staticdir.on": True,
            "tools.staticdir.dir": "./"
        }
    }
    cherrypy.quickstart(
        Root(root_dir, args.thumbnail_size, args.rotate), "/", conf)


if __name__ == "__main__":
    main()
