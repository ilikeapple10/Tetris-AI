import argparse
import cv2
import torch
from tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()

    env = Tetris(height=opt.height, width=opt.width, block_size=opt.block_size)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()

    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"XVID"), opt.fps,
                    (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))

    
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())

        if torch.cuda.is_available():
            next_states = next_states.cuda()

        next_states_tensor = torch.stack(next_states)
        predictions = model(next_states_tensor)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break
    
if __name__ == "__main__":
    opt = get_args()
    test(opt)