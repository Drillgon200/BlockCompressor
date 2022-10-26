# BlockCompressor
This project mostly consists of my experiments with texture compression. BC7 is the main one I care about, so I put at least some effort into making it good (though it's definitely not going to beat the existing compressors). Currently, the BC7 compressor uses AVX2 intrinsics to compress 8 blocks at a time, as well as using a job system I wrote a while ago to multithread it.

The quality is actually pretty good (at least from my very low amount of testing). By a simple linear PSNR metric (I know this isn't the best, and I'd like to implement some kind of visual error weighting at some point), it comes pretty close to Nvidia's texture tools for the couple of images I've tested.
By the eyeball test, it's pretty hard to tell the difference between compressed and uncompressed images, and that's good enough for me.

Along with the compression stuff, it also contains my first attempt at writing a PNG loader because I wanted to learn how to do that.

This project is designed to be a prototype space for texture stuff in my StarChicken game engine.