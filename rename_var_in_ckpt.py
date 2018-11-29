import tensorflow as tf
import os

if __name__ == "__main__":
    checkpoint_dir = os.path.join("/home/ky/work/something", "Test")
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = "t1/" + var_name

            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)


        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join("/home/ky/work/something", "Test2"))
