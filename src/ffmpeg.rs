use std::collections::HashMap;

use ffmpeg_sidecar::paths::sidecar_dir;
use ffmpeg_sidecar::{
    command::{ffmpeg_is_installed, FfmpegCommand},
    event::{FfmpegEvent, FfmpegProgress},
};
use indicatif::{ProgressBar, ProgressStyle};

pub fn ensure_ffmpeg(verbose: bool) {
    if !ffmpeg_is_installed() {
        if verbose {
            println!(
                "No ffmpeg installation found, downloading one to {}...",
                &sidecar_dir().unwrap().display()
            );
        }
        ffmpeg_sidecar::download::auto_download().unwrap();
    }
}

pub fn make_video(
    pattern: &str,
    outfile: &str,
    fps: u64,
    num_frames: u64,
    pbar_style: Option<ProgressStyle>,
    metadata: Option<HashMap<&str, &str>>,
) {
    let pbar = if let Some(style) = pbar_style {
        ProgressBar::new(num_frames).with_style(style)
    } else {
        ProgressBar::hidden()
    };

    let metadata_args = metadata
        .map(|map| {
            let mut s: Vec<String> = Vec::new();
            for (key, val) in map.iter() {
                s.push("-metadata".to_string());
                s.push(format!("{key}={val}"));
            }
            s
        })
        .unwrap_or(vec!["".to_string()]);

    let cmd = format!(
        // Scale to a max width of 1280 pixels as long as the height is divisible by 2
        "-framerate {fps} -f image2 -i {pattern} -y -vcodec libx264 -crf 22 -pix_fmt yuv420p -vf scale=1280:-2"
    );
    let mut output = "".to_owned();

    let mut ffmpeg_runner = FfmpegCommand::new()
        .args(cmd.split(' '))
        .args(metadata_args)
        .output(outfile)
        // .print_command()
        .spawn()
        .unwrap();

    ffmpeg_runner.iter().unwrap().for_each(|e| match e {
        FfmpegEvent::Progress(FfmpegProgress { frame, .. }) => pbar.set_position(frame as u64),
        FfmpegEvent::Log(_level, msg) => {
            if !msg.is_empty() {
                output.push_str(&format!("[ffmpeg] {msg}\n"))
            }
        }
        _ => {}
    });
    pbar.finish_and_clear();

    if !ffmpeg_runner.wait().unwrap().success() {
        println!("FFMPEG Failed.");
        println!("Command: ffmpeg {cmd}");
        print!("{output}");
    }
}
