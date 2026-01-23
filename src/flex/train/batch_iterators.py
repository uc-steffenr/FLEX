"""Defines batch iteration methods."""
import os
import numpy as np
import jax.numpy as jnp


def data_batch_iterator(
        X,
        Y,
        batch_size,
        *,
        infinite=True,
        fill_batch=False,
):
    rng = np.random.default_rng(12345)

    xb_buf = None
    yb_buf = None
    buf_n = 0

    while True:
        idx = rng.permutation(len(X))
        X = X[idx]
        Y = Y[idx]

        n = len(X)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            x_np = X[start:end]
            y_np = Y[start:end]
            m = end - start

            if not fill_batch:
                yield jnp.array(x_np), jnp.array(y_np)

            if xb_buf is None:
                xb_buf = x_np
                yb_buf = y_np
                buf_n = m
            else:
                xb_buf = np.concatenate([xb_buf, x_np], axis=0)
                yb_buf = np.concatenate([yb_buf, y_np], axis=0)
                buf_n += m

            while buf_n >= batch_size:
                xb = jnp.array(xb_buf[:batch_size])
                yb = jnp.array(yb_buf[:batch_size])
                yield xb, yb

                # keep leftovers in buffer
                if buf_n == batch_size:
                    xb_buf = None
                    yb_buf = None
                    buf_n = 0
                else:
                    xb_buf = xb_buf[batch_size:]
                    yb_buf = yb_buf[batch_size:]
                    buf_n -= batch_size

        if not infinite:
            break

    if fill_batch and buf_n > 0:
        yield jnp.array(xb_buf), jnp.array(yb_buf)


def shard_batch_iterator(
        dir,
        batch_size,
        *,
        infinite=True,
        fill_batch=False,
):
    paths = sorted(
        os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".npz")
    )
    rng = np.random.default_rng(12345)

    xb_buf = None
    yb_buf = None
    buf_n = 0

    while True:
        rng.shuffle(paths)
        for path in paths:
            data = np.load(path)
            X = data["X"]
            Y = data["Y"]

            idx = rng.permutation(len(X))
            X = X[idx]
            Y = Y[idx]

            n = len(X)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                x_np = X[start:end]
                y_np = Y[start:end]
                m = end - start

                if not fill_batch:
                    yield jnp.array(x_np), jnp.array(y_np)

                if xb_buf is None:
                    xb_buf = x_np
                    yb_buf = y_np
                    buf_n = m
                else:
                    xb_buf = np.concatenate([xb_buf, x_np], axis=0)
                    yb_buf = np.concatenate([yb_buf, y_np], axis=0)
                    buf_n += m

                while buf_n >= batch_size:
                    xb = jnp.array(xb_buf[:batch_size])
                    yb = jnp.array(yb_buf[:batch_size])
                    yield xb, yb

                    # keep leftovers in buffer
                    if buf_n == batch_size:
                        xb_buf = None
                        yb_buf = None
                        buf_n = 0
                    else:
                        xb_buf = xb_buf[batch_size:]
                        yb_buf = yb_buf[batch_size:]
                        buf_n -= batch_size

        if not infinite:
            break

    if fill_batch and buf_n > 0:
        yield jnp.array(xb_buf), jnp.array(yb_buf)

