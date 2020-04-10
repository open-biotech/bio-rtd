import unittest
import io
import sys

from bio_rtd import logger


class DummyUO:
    _log_entity_id = "DummyUnitOp"
    default_logger = logger.DefaultLogger()
    data_storing_logger = logger.DataStoringLogger()
    strict_logger = logger.StrictLogger()

    def write_to_default_logger(self, lvl, msg):
        self.default_logger.log(lvl, msg)

    def write_to_data_storing_logger(self, lvl, msg):
        self.data_storing_logger.log(lvl, msg)

    def write_to_strict_logger(self, lvl, msg):
        self.strict_logger.log(lvl, msg)

    def data_write_to_data_storing_logger(self, tree, lvl, key, value):
        if lvl == self.data_storing_logger.INFO:
            self.data_storing_logger.i_data(tree, key, value)
        else:
            self.data_storing_logger.d_data(tree, key, value)


class TestLogger(unittest.TestCase):

    def setUp(self):
        # Print to internal output.
        self.old_stdout = sys.stdout
        sys.stdout = self.print_output = io.StringIO()
        # empty function is there just to compare it to other empty functions
        self.empty_func()

    def tearDown(self):
        # restore print
        sys.stdout = self.old_stdout

    def assert_print(self, msg: str, match=True):
        if match:
            self.assertEqual(msg, self.print_output.getvalue().split('\n')[-2])
        else:
            self.assertNotEqual(msg,
                                self.print_output.getvalue().split('\n')[-2])

    @staticmethod
    def empty_func():
        pass

    def assert_empty_function(self, fun):
        self.assertEqual(fun.__code__.co_code,
                         self.empty_func.__code__.co_code)

    def test_default_logger(self):
        log = logger.DefaultLogger()

        # default values
        self.assertEqual(log.log_level, log.WARNING)
        self.assertEqual(log.log_data, False)

        # test error
        with self.assertRaises(RuntimeError, msg="Error1"):
            log.e("Error1")
        # test print
        log.w("Print warning 1")
        self.assert_print("WARNING: Print warning 1")
        # test print from unit op
        uo = DummyUO()
        uo.write_to_default_logger(log.WARNING, "Print warning 2")
        self.assert_print("WARNING: DummyUnitOp: Print warning 2")

        # test levels
        def test_log_lvl(min_lvl, max_lvl):
            for log_lvl in (log.ERROR, log.WARNING, log.INFO, log.DEBUG):
                msg = f"Print at log_lvl {log_lvl}"
                target_msg = f"WARNING: {msg}" if log_lvl == log.WARNING \
                    else msg
                if log_lvl <= max_lvl:
                    log.log(log_lvl, msg)
                    self.assert_print(target_msg, match=log_lvl >= min_lvl)

        log.log_level = log.WARNING
        test_log_lvl(min_lvl=log.WARNING, max_lvl=log.WARNING)
        log.log_level = log.INFO
        test_log_lvl(min_lvl=log.INFO, max_lvl=log.WARNING)
        log.log_level = log.DEBUG
        test_log_lvl(min_lvl=log.DEBUG, max_lvl=log.WARNING)

        # skip adding any data
        data_tree = dict()
        log.set_data_tree("MainTree", data_tree)
        log.d_data(data_tree, "bo1", 10)
        log.i_data(data_tree, "bo2", 20)
        self.assertTrue(len(data_tree.keys()) == 0)

        # enable adding data
        log.log_data = True
        data_tree = dict()
        log.set_data_tree("MainTree2", data_tree)
        log.d_data(data_tree, "ab1", 11)
        log.i_data(data_tree, "ab2", 22)
        ref_dict = {"ab1": 11, "ab2": 22}
        self.assertTrue(ref_dict == log.get_data_tree("MainTree2"))
        self.assertTrue(ref_dict == data_tree)
        self.assertTrue(data_tree == log.get_entire_data_tree()["MainTree2"])

        # branch tree
        branch_tree = dict()
        log.set_branch(data_tree, "BranchTree", branch_tree)
        log.i_data(branch_tree, "cab1", 122)
        ref_dict = {"ab1": 11, "ab2": 22, "BranchTree": {"cab1": 122}}
        self.assertTrue(ref_dict == log.get_data_tree("MainTree2"))

        # branch list
        branch_list = list()
        log.set_branch(data_tree, "BranchList", branch_list)
        branch_list += [1, 2, 3]
        ref_dict = {"ab1": 11,
                    "ab2": 22,
                    "BranchTree": {"cab1": 122},
                    "BranchList": [1, 2, 3]}
        self.assertTrue(ref_dict == log.get_data_tree("MainTree2"))

    def test_data_storing_logger(self):
        log = logger.DataStoringLogger()

        # default values
        self.assertEqual(log.log_level, log.WARNING)
        self.assertEqual(log.log_data, True)

        # test print
        log.log_level = log.ERROR
        log.e("Print error 1")
        self.assert_print("ERROR: Print error 1")
        log.w("Print warning 1")  # ignore due to log_level
        self.assert_print("WARNING: Print warning 1", match=False)
        log.log_level = log.WARNING
        log.e("Print error 2")
        self.assert_print("ERROR: Print error 2")
        log.w("Print warning 2")
        self.assert_print("WARNING: Print warning 2")
        log.i("Print info 1")  # ignore due to log_level
        self.assert_print("INFO: Print info 1", match=False)
        log.log_level = log.INFO
        log.i("Print info 2")
        self.assert_print("INFO: Print info 2")
        log.log(log.DEBUG, "Print debug 1")  # ignore due to log_level
        self.assert_print("DEBUG: Print debug 1", match=False)
        log.log_level = log.DEBUG
        log.log(log.DEBUG, "Print debug 2")
        self.assert_print("DEBUG: Print debug 2")
        # test print from unit op
        uo = DummyUO()
        uo.write_to_data_storing_logger(log.WARNING, "Print warning 2")
        self.assert_print("WARNING: DummyUnitOp: Print warning 2")

        # test adding data
        data_tree = dict()
        log.log_level = log.INFO
        log.log_level_data = log.DEBUG
        log.set_data_tree("MainTree3", data_tree)
        log.d_data(data_tree, "ab1", 11)
        self.assert_print("DEBUG: Value set: ab1: 11", match=False)
        log.i_data(data_tree, "ab2", 22)
        self.assert_print("INFO: Value set: ab2: 22")
        ref_dict = {"ab1": 11, "ab2": 22}
        self.assertEqual(ref_dict, log.get_data_tree("MainTree3"))
        self.assertEqual(ref_dict, log.get_data_tree("MainTree3"))
        log.i_data(data_tree, "ab3", 33)
        self.assert_print("INFO: Value set: ab3: 33")

        # test adding data in unit op
        data_tree = dict()
        uo.data_storing_logger.set_data_tree("MainTree4", data_tree)
        uo.data_storing_logger.log_level = uo.data_storing_logger.INFO
        uo.data_write_to_data_storing_logger(data_tree, log.DEBUG, "ac1", 12)
        self.assert_print("DEBUG: DummyUnitOp: Value set: ac1: 12",
                          match=False)
        uo.data_write_to_data_storing_logger(data_tree, log.INFO, "ac2", 23)
        self.assert_print("INFO: DummyUnitOp: Value set: ac2: 23")
        ref_dict = {"ac1": 12, "ac2": 23}
        self.assertEqual(ref_dict, data_tree)
        self.assertEqual(ref_dict,
                         uo.data_storing_logger.get_data_tree("MainTree4"))
        uo.data_write_to_data_storing_logger(data_tree, log.INFO, "ac3", 34)
        self.assert_print("INFO: DummyUnitOp: Value set: ac3: 34")

    def test_strict_logger(self):
        log = logger.StrictLogger()

        # default values
        self.assertEqual(log.log_level, log.WARNING)
        self.assertEqual(log.log_data, False)

        # skip info and debug
        log.i("Nothing - Info")

        # Errors on warning and above
        with self.assertRaises(RuntimeError,
                               msg="w triggered RuntimeError via logger"):
            log.w("w triggered RuntimeError via logger")
        with self.assertRaises(RuntimeError,
                               msg="e triggered RuntimeError via logger"):
            log.e("e triggered RuntimeError via logger")

        # Uo prefix
        uo = DummyUO()
        with self.assertRaises(RuntimeError, msg="w triggered RuntimeError"):
            uo.write_to_strict_logger(
                log.WARNING,
                "DummyUnitOp: w triggered RuntimeError"
            )

        # skip adding any data
        data_tree = dict()
        log.set_data_tree("MainTree", data_tree)
        log.d_data(data_tree, "bo1", 10)
        log.i_data(data_tree, "bo2", 20)
        self.assertTrue(len(data_tree.keys()) == 0)

        # ensure empty function and call function for 100 % coverage
        self.assert_empty_function(log._on_data_stored)
        log._on_data_stored(0, dict(), "", "")
