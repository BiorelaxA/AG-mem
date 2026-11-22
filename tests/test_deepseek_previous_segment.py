import unittest
from tests.test_adjacent_writer import TOKENIZER, Transport, session
from agmem.build_deepseek_previous_segment import PreviousSegmentWriter


class PreviousSegmentTests(unittest.TestCase):
    def test_prior_segment_is_complete_and_is_separate_from_current_facts(self):
        transport = Transport()
        writer = PreviousSegmentWriter(TOKENIZER, {"semantic_retries": 0, "context_blocks": 0}, transport)
        spans = [{"start_block": 1, "end_block": 3}, {"start_block": 4, "end_block": 5},
                 {"start_block": 6, "end_block": 7}]
        writer.build(session(7), spans)
        children = [call[4] for call in transport.calls if call[3] == "adjacent_child"]
        self.assertEqual(len(children), 3)
        self.assertEqual(children[0]["context_segments"], [])
        for i in (1, 2):
            self.assertEqual(children[i]["context_segments"], [{**spans[i-1], "blocks": children[i-1]["blocks"]}])
            self.assertNotIn("context_blocks", children[i])
        self.assertEqual(transport.calls[-1][3], "adjacent_parent")
        self.assertNotIn("context_segments", transport.calls[-1][4])

    def test_portable_import_from_original_then_v3(self):
        from tests.test_portable_deepseek_migration import PortableDeepSeekMigrationTests
        from agmem import build_deepseek_previous_segment as mod
        from agmem import build_portable_deepseek_directories as portable

        fixture = PortableDeepSeekMigrationTests(methodName="runTest")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        portable.run(fixture.args("--import-only"), tokenizer=TOKENIZER)
        old_target = fixture.target
        fixture.config.update(protocol=mod.PROTOCOL, child_context_segments=1)
        fixture.config["writer"]["context_blocks"] = 0
        for source in (fixture.source, old_target):
            output = fixture.project / ("new-" + source.name)
            fixture.config["output_root"] = str(output)
            fixture.write_config()
            args = fixture.args("--source-root", str(source), "--import-only")
            with portable.portable_worker(fixture.manifest_path, fixture.recipe) as proof, mod.previous_segment_worker(proof):
                report = mod.v3.run(args, tokenizer=TOKENIZER, writer_factory=mod.PreviousSegmentWriter)
            self.assertEqual(report["counters"]["imported_complete"], 1)
            self.assertEqual(report["target_identity"]["child_context_policy"]["count"], 1)
